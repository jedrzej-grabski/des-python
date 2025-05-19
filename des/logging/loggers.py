from typing import Any, final, Optional
import numpy as np
from numpy.typing import NDArray
from des.config import DESConfig


@final
class LogData:
    """Container for diagnostic log data."""

    def __init__(self) -> None:
        """Initialize empty log containers."""
        self.Ft: list[float] = []
        self.value: list[NDArray[np.float64]] = []
        self.mean: list[float] = []
        self.meanCords: list[NDArray[np.float64]] = []
        self.pop: list[NDArray[np.float64]] = []
        self.bestVal: list[float] = []
        self.worstVal: list[float] = []
        self.eigen: list[NDArray[np.float64]] = []

    def clear(self) -> None:
        """Reset all log data."""
        self.Ft = []
        self.value = []
        self.mean = []
        self.meanCords = []
        self.pop = []
        self.bestVal = []
        self.worstVal = []
        self.eigen = []

    def to_dict(self) -> dict[str, list[Any]]:
        """Convert log data to dictionary format."""
        return {
            "Ft": self.Ft,
            "value": self.value,
            "mean": self.mean,
            "meanCords": self.meanCords,
            "pop": self.pop,
            "bestVal": self.bestVal,
            "worstVal": self.worstVal,
            "eigen": self.eigen,
        }


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
        self.logs = LogData()

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
            self.logs.Ft.append(ft)

        if self.config.diag_value:
            self.logs.value.append(fitness.copy())

        if self.config.diag_mean:
            self.logs.mean.append(mean_fitness)

        if self.config.diag_meanCords:
            self.logs.meanCords.append(mean_coords.copy())

        if self.config.diag_pop:
            self.logs.pop.append(population.copy())

        if self.config.diag_bestVal:
            self.logs.bestVal.append(best_fitness)

        if self.config.diag_worstVal:
            self.logs.worstVal.append(worst_fitness)

        if self.config.diag_eigen:
            self.logs.eigen.append(eigen_values.copy())

    def get_logs(self) -> LogData:
        """
        Get all logged data.

        Returns:
            Object containing logged diagnostic information
        """
        return self.logs

    def get_logs_dict(self) -> dict[str, Any]:
        """
        Get all logged data as a dictionary.

        Returns:
            Dictionary with logged diagnostic information
        """
        return self.logs.to_dict()

    def clear_logs(self) -> None:
        """Clear all logged data."""
        self.logs.clear()
