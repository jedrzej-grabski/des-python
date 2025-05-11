from typing import Any, final
import numpy as np
from numpy.typing import NDArray
from des.config import DESConfig


@final
class DiagnosticLogger:
    """
    Logger for capturing diagnostic information during optimization.
    """

    def __init__(
        self, config: DESConfig, dimensions: int, max_iter: int, population_size: int
    ) -> None:
        """
        Initialize the diagnostic logger.

        Args:
            config: DES configuration object
            dimensions: Number of dimensions
            max_iter: Maximum number of iterations
            population_size: Population size (lambda)
        """
        self.config = config
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.population_size = population_size

        # Initialize data storage
        self.logs: dict[str, list[Any]] = {}

        # Setup log storage based on enabled diagnostics
        if config.diag_Ft:
            self.logs["Ft"] = []

        if config.diag_value:
            self.logs["value"] = []

        if config.diag_mean:
            self.logs["mean"] = []

        if config.diag_meanCords:
            self.logs["meanCords"] = []

        if config.diag_pop:
            self.logs["pop"] = []

        if config.diag_bestVal:
            self.logs["bestVal"] = []

        if config.diag_worstVal:
            self.logs["worstVal"] = []

        if config.diag_eigen:
            self.logs["eigen"] = []

    def log_iteration(
        self,
        ft: float,
        fitness: NDArray[np.float64],
        mean_fitness: float,
        mean_coords: NDArray[np.float64],
        population: NDArray[np.float64],
        best_fitness: float,
        worst_fitness: float,
        eigen_values: NDArray[np.float64],
    ) -> None:
        """
        Log diagnostic information for the current iteration.

        Args:
            ft: Current Ft value
            fitness: Array of fitness values for the population
            mean_fitness: Fitness of the mean point
            mean_coords: Mean point coordinates
            population: Current population
            best_fitness: Best fitness in the population
            worst_fitness: Worst fitness in the population
            eigen_values: Eigenvalues of the population covariance matrix
        """

        if self.config.diag_Ft:
            self.logs["Ft"].append(ft)

        if self.config.diag_value:
            self.logs["value"].append(fitness.copy())

        if self.config.diag_mean:
            self.logs["mean"].append(mean_fitness)

        if self.config.diag_meanCords:
            self.logs["meanCords"].append(mean_coords.copy())

        if self.config.diag_pop:
            self.logs["pop"].append(population.copy())

        if self.config.diag_bestVal:
            self.logs["bestVal"].append(best_fitness)

        if self.config.diag_worstVal:
            self.logs["worstVal"].append(worst_fitness)

        if self.config.diag_eigen:
            self.logs["eigen"].append(eigen_values.copy())

    def get_logs(self) -> dict[str, Any]:
        """
        Get all logged data.

        Returns:
            Dictionary with logged diagnostic information
        """
        return self.logs

    def clear_logs(self) -> None:
        """Clear all logged data."""
        for key in self.logs:
            self.logs[key] = []
