from typing import Any, Callable, final
from dataclasses import dataclass
from scipy.special import gamma
import numpy as np
from numpy.typing import NDArray
import math

from des.utils.ring_buffer import RingBuffer
from des.utils.boundary_handlers import (
    create_boundary_handler,
    BoundaryHandler,
    BoundaryHandlerType,
)
from des.utils.helpers import (
    norm,
    success_probability,
    calculate_ft,
    delete_inf_nan,
)
from des.logging.loggers import DiagnosticLogger
from des.config import DESConfig


@dataclass
class OptimizationResult:
    """
    Result of an optimization run.
    """

    best_solution: NDArray[np.float64]
    best_fitness: float
    evaluations: int
    resets: int
    convergence: int  # 0: success, 1: maxiter reached
    message: str
    diagnostic: dict[str, Any]


@final
class DESOptimizer:
    """
    Differential Evolution with Success-History Based Parameter Adaptation.
    """

    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: DESConfig | None = None,
        lower_bounds: float | NDArray[np.float64] | list[float] = -100.0,
        upper_bounds: float | NDArray[np.float64] | list[float] = 100.0,
        boundary_strategy: BoundaryHandlerType = BoundaryHandlerType.BOUNCE_BACK,
    ) -> None:
        """
        Initialize the DES optimizer.

        Args:
            func: Objective function to minimize
            initial_point: Initial guess for the solution
            lower_bounds: Lower bounds for each dimension or a single value for all dimensions
            upper_bounds: Upper bounds for each dimension or a single value for all dimensions
            config: Configuration object for the optimizer
            boundary_strategy: Strategy for handling boundary constraints ("bounce_back" or "clamp")
        """
        self.func = func
        self.initial_point = np.array(initial_point, dtype=float)
        self.dimensions = len(initial_point)
        self.evaluations = 0

        # Process bounds
        if isinstance(lower_bounds, (int, float)):
            self.lower_bounds = np.full(self.dimensions, lower_bounds)
        else:
            self.lower_bounds = np.array(lower_bounds, dtype=float)

        if isinstance(upper_bounds, (int, float)):
            self.upper_bounds = np.full(self.dimensions, upper_bounds)
        else:
            self.upper_bounds = np.array(upper_bounds, dtype=float)

        # Create boundary handler
        self.boundary_handler = create_boundary_handler(
            boundary_strategy, self.lower_bounds, self.upper_bounds
        )

        # Initialize configuration
        self.config = config if config is not None else DESConfig(self.dimensions)

    def optimize(self) -> OptimizationResult:
        """
        Run the optimization algorithm.

        Returns:
            OptimizationResult object containing optimization results
        """
        # Initialize parameters
        N = self.dimensions
        budget = self.config.budget
        lambda_ = self.config.population_size
        minlambda = self.config.minlambda
        pathLength = self.config.pathLength
        initFt = self.config.initFt
        stopfitness = self.config.stopfitness
        histSize = self.config.history
        c_Ft = self.config.c_Ft
        cp = self.config.cp
        tol = self.config.tol
        max_iter = self.config.maxit
        lamarckism = self.config.Lamarckism
        weights = self.config.weights
        mu = self.config.mu
        mueff = self.config.mueff
        ccum = self.config.ccum
        pathRatio = self.config.pathRatio

        # Initialize optimization variables
        self.evaluations = 0
        best_fitness = float("inf")
        best_solution = self.initial_point.copy()
        worst_fitness = None
        message = None
        restart_number = -1
        iter_count = 0

        # Initialize logger
        logger = self._create_logger(N, max_iter, lambda_)

        # Run optimization
        while self.evaluations < budget:
            restart_number += 1
            mu = math.floor(lambda_ / 2)
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights = weights / np.sum(weights)
            weights_pop = np.log(lambda_ + 1) - np.log(np.arange(1, lambda_ + 1))
            weights_pop = weights_pop / np.sum(weights_pop)

            # Initialize evolution parameters
            hist_head = 0
            history: list[NDArray[np.float64]] = []
            Ft = initFt

            # Create first population
            population = np.random.uniform(
                0.8 * self.lower_bounds[:, None],
                0.8 * self.upper_bounds[:, None],
                size=(N, lambda_),
            )
            cum_mean = (self.upper_bounds + self.lower_bounds) / 2
            population_repaired = np.apply_along_axis(
                self.boundary_handler.repair, 0, population
            )

            if lamarckism:
                population = population_repaired

            # Evaluate initial population
            fitness = self._evaluate_population(
                population if lamarckism else population_repaired
            )

            old_mean = np.zeros(N)
            new_mean = self.initial_point.copy()
            worst_fitness = np.max(fitness)

            # Store population and selection means
            pop_mean = np.sum(population * weights_pop.reshape(1, -1), axis=1)
            mu_mean = new_mean

            # Initialize matrices for creating diffs
            diffs = np.zeros((N, lambda_))

            # Calculate chi_N
            chi_N = np.sqrt(2) * gamma((N + 1) / 2) / gamma(N / 2)
            hist_norm = 1 / np.sqrt(2)
            counter_repaired = 0

            # Allocate buffers
            steps = RingBuffer(pathLength * N)
            d_mean = np.zeros((N, histSize))
            ft_history = np.zeros(histSize)
            pc = np.zeros((N, histSize))

            stop_tol = False
            while self.evaluations < budget and not stop_tol:
                iter_count += 1
                hist_head = (hist_head % histSize) + 1

                mu = math.floor(lambda_ / 2)
                weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
                weights = weights / np.sum(weights)

                # Log diagnostic information
                logger.log_iteration(
                    ft=Ft,
                    fitness=fitness,
                    mean_fitness=self._evaluate(self.boundary_handler.repair(new_mean)),
                    mean_coords=new_mean,
                    population=population,
                    best_fitness=np.min(fitness),
                    worst_fitness=np.max(fitness),
                    eigen_values=np.sort(np.linalg.eigvals(np.cov(population)))[::-1],
                )

                # Select best mu individuals
                selection = np.argsort(fitness)[:mu]
                selected_points = population[:, selection]

                # Save selected population in history buffer
                if len(history) < histSize:
                    history.append(selected_points * hist_norm / Ft)
                else:
                    history[hist_head - 1] = selected_points * hist_norm / Ft

                # Calculate weighted mean of selected points
                old_mean = new_mean.copy()
                new_mean = np.sum(selected_points * weights.reshape(1, -1), axis=1)

                # Write to buffers
                mu_mean = new_mean
                d_mean[:, hist_head - 1] = (mu_mean - pop_mean) / Ft

                step = (new_mean - old_mean) / Ft

                # Update buffer of steps
                steps.push_all(step)

                # Update Ft
                ft_history[hist_head - 1] = Ft
                if (
                    iter_count > pathLength - 1
                    and not np.any(step == 0)
                    and counter_repaired < 0.1 * lambda_
                ):
                    Ft = calculate_ft(
                        steps.peek(),
                        N,
                        lambda_,
                        pathLength,
                        Ft,
                        c_Ft,
                        pathRatio,
                        chi_N,
                        mueff,
                    )

                # Update parameters
                if hist_head == 1:
                    pc[:, hist_head - 1] = (1 - cp) * np.zeros(N) / np.sqrt(
                        N
                    ) + np.sqrt(mu * cp * (2 - cp)) * step
                else:
                    pc[:, hist_head - 1] = (1 - cp) * pc[:, hist_head - 2] + np.sqrt(
                        mu * cp * (2 - cp)
                    ) * step

                # Sample from history
                limit = min(iter_count, histSize)
                history_sample = np.random.choice(range(limit), lambda_, replace=True)
                history_sample2 = np.random.choice(range(limit), lambda_, replace=True)

                x1_sample = np.zeros(lambda_, dtype=int)
                x2_sample = np.zeros(lambda_, dtype=int)

                for i in range(lambda_):
                    hist_idx = history_sample[i]
                    x1_sample[i] = np.random.randint(0, history[hist_idx].shape[1])
                    x2_sample[i] = np.random.randint(0, history[hist_idx].shape[1])

                # Make diffs
                for i in range(lambda_):
                    hist_idx = history_sample[i]
                    x1 = history[hist_idx][:, x1_sample[i]]
                    x2 = history[hist_idx][:, x2_sample[i]]

                    diffs[:, i] = (
                        np.sqrt(ccum)
                        * (x1 - x2 + np.random.normal() * d_mean[:, hist_idx])
                        + np.sqrt(1 - ccum)
                        * np.random.normal()
                        * pc[:, history_sample2[i]]
                    )

                # Generate new population
                population = (
                    new_mean.reshape(-1, 1)
                    + Ft * diffs
                    + tol
                    * (1 - 2 / N**2) ** (iter_count / 2)
                    * np.random.normal(size=diffs.shape)
                    / chi_N
                )

                population = delete_inf_nan(population)

                # Check constraints violations and repair if necessary
                population_temp = population.copy()
                population_repaired = np.apply_along_axis(
                    self.boundary_handler.repair, 0, population
                )

                # Count repaired individuals
                counter_repaired = 0
                for i in range(population.shape[1]):
                    if not np.array_equal(
                        population_temp[:, i], population_repaired[:, i]
                    ):
                        counter_repaired += 1

                if lamarckism:
                    population = population_repaired

                pop_mean = np.sum(population * weights_pop.reshape(1, -1), axis=1)

                # Evaluate population
                fitness = self._evaluate_population(
                    population if lamarckism else population_repaired
                )

                fitness_non_lamarckian = None
                if not lamarckism:
                    fitness_non_lamarckian = self._apply_penalty(
                        population, population_repaired, fitness, worst_fitness
                    )

                # Check for best fitness
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < best_fitness:
                    best_fitness = fitness[best_idx]
                    best_solution = (
                        population[:, best_idx]
                        if lamarckism
                        else population_repaired[:, best_idx]
                    )

                # Check worst fitness
                worst_idx = np.argmax(fitness)
                if fitness[worst_idx] > worst_fitness:
                    worst_fitness = fitness[worst_idx]

                # Apply penalty for non-Lamarckian approach
                if not lamarckism:
                    assert fitness_non_lamarckian is not None
                    fitness = fitness_non_lamarckian

                # Check if the mean point is better
                cum_mean = 0.8 * cum_mean + 0.2 * new_mean
                cum_mean_repaired = self.boundary_handler.repair(cum_mean)
                mean_fitness = self._evaluate(cum_mean_repaired)

                if mean_fitness < best_fitness:
                    best_fitness = mean_fitness
                    best_solution = cum_mean_repaired

                # Check termination conditions
                if fitness[0] <= stopfitness:
                    message = "Stop fitness reached."
                    break

                # Termination if fitness range is very small and most of budget used
                if (
                    abs(np.max(fitness) - np.min(fitness)) < tol
                    and self.evaluations > 0.8 * budget
                ):
                    stop_tol = True

        # Create result object
        result = OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations=self.evaluations,
            resets=restart_number,
            convergence=1 if iter_count >= max_iter else 0,
            message=message if message else "Maximum function evaluations reached.",
            diagnostic=logger.get_logs(),
        )

        return result

    def _create_logger(
        self, dimensions: int, max_iter: int, population_size: int
    ) -> DiagnosticLogger:
        """
        Create a diagnostic logger configured based on the optimizer's config.

        Args:
            dimensions: Number of dimensions
            max_iter: Maximum number of iterations
            population_size: Population size

        Returns:
            Configured DiagnosticLogger instance
        """
        return DiagnosticLogger(self.config, dimensions, max_iter, population_size)

    def _evaluate(self, x: NDArray[np.float64]) -> float:
        """
        Evaluate a single solution.

        Args:
            x: Solution to evaluate

        Returns:
            Fitness value
        """
        if self.boundary_handler.is_feasible(x):
            self.evaluations += 1
            return self.func(x)
        else:
            return float("inf")

    def _evaluate_population(
        self, population: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Evaluate a population of solutions.

        Args:
            population: Matrix of solution vectors (each column is a solution)

        Returns:
            Array of fitness values
        """
        fitness = np.zeros(population.shape[1])
        budget_left = self.config.budget - self.evaluations

        if budget_left >= population.shape[1]:
            # Evaluate entire population
            for i in range(population.shape[1]):
                fitness[i] = self._evaluate(population[:, i])
        else:
            # Evaluate only what budget allows
            for i in range(budget_left):
                fitness[i] = self._evaluate(population[:, i])

            # Fill rest with infinity
            fitness[budget_left:] = float("inf")

        return fitness

    def _apply_penalty(
        self,
        population: NDArray[np.float64],
        repaired_population: NDArray[np.float64],
        fitness: NDArray[np.float64],
        worst_fitness: float,
    ) -> NDArray[np.float64]:
        """
        Apply penalty to solutions that violated constraints.

        Args:
            population: Original population
            repaired_population: Repaired population
            fitness: Fitness values of repaired population
            worst_fitness: Worst fitness value seen so far

        Returns:
            Penalized fitness values
        """
        # Find individuals that needed repair
        needs_repair = np.zeros(population.shape[1], dtype=bool)
        for i in range(population.shape[1]):
            needs_repair[i] = not np.array_equal(
                population[:, i], repaired_population[:, i]
            )

        # Calculate squared distances between original and repaired
        sq_distances = np.zeros(population.shape[1])
        for i in range(population.shape[1]):
            if needs_repair[i]:
                sq_distances[i] = np.sum(
                    (population[:, i] - repaired_population[:, i]) ** 2
                )

        # Apply penalty based on distance
        penalized_fitness = fitness.copy()
        penalized_fitness[needs_repair] = worst_fitness + sq_distances[needs_repair]

        return delete_inf_nan(penalized_fitness)
