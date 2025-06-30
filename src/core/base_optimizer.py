from abc import ABC, abstractmethod
from typing import Callable, Union, TypeVar, Generic, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from src.algorithms.choices import AlgorithmChoice
from src.logging.base_logger import BaseLogData, BaseLogger

from src.logging.logger_factory import LoggerFactory
from src.core.config_base import BaseConfig
from src.utils.boundary_handlers import (
    BoundaryHandler,
    create_boundary_handler,
    BoundaryHandlerType,
)


# Type variables
LogDataType = TypeVar("LogDataType", bound=BaseLogData)
ConfigType = TypeVar("ConfigType", bound=BaseConfig)


@dataclass
class OptimizationResult(Generic[LogDataType]):
    """Result of an optimization run with proper typing."""

    best_solution: NDArray[np.float64]
    best_fitness: float
    evaluations: int
    message: str
    diagnostic: LogDataType
    algorithm: AlgorithmChoice = AlgorithmChoice.Unknown


class BaseOptimizer(ABC, Generic[LogDataType, ConfigType]):
    """Abstract base class for optimization algorithms."""

    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: ConfigType,
        algorithm: AlgorithmChoice = AlgorithmChoice.Unknown,
        boundary_handler: BoundaryHandler | None = None,
        boundary_strategy: BoundaryHandlerType | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
    ) -> None:
        """Initialize the base optimizer."""
        self.func = func
        self.initial_point = np.array(initial_point, dtype=float)
        self.dimensions = len(initial_point)
        self.config: ConfigType = config  # Now properly typed!
        self.algorithm = algorithm
        self.evaluations = 0

        # Process bounds and set up boundary handler
        self.lower_bounds = self._process_bounds(lower_bounds, self.dimensions)
        self.upper_bounds = self._process_bounds(upper_bounds, self.dimensions)
        self._validate_bounds()

        if boundary_handler is not None:
            self.boundary_handler = boundary_handler
        else:
            boundary_strategy = boundary_strategy or BoundaryHandlerType.CLAMP
            self.boundary_handler = create_boundary_handler(
                boundary_strategy, self.lower_bounds, self.upper_bounds
            )

        # Initialize logger using factory
        self.logger: BaseLogger[LogDataType] = LoggerFactory.create_logger(
            algorithm, config
        )

    @staticmethod
    def _process_bounds(
        bounds: Union[float, NDArray[np.float64], list[float]], dimensions: int
    ) -> NDArray[np.float64]:
        """Process bounds input into numpy array format."""
        if isinstance(bounds, (int, float)):
            return np.full(dimensions, bounds, dtype=float)
        else:
            return np.array(bounds, dtype=float)

    def _validate_bounds(self) -> None:
        """Validate that bounds are compatible."""
        if self.lower_bounds.shape != self.upper_bounds.shape:
            raise ValueError("Lower and upper bounds must have the same shape.")
        if np.any(self.lower_bounds > self.upper_bounds):
            raise ValueError("Lower bounds must be less than or equal to upper bounds.")

    def evaluate(self, x: NDArray[np.float64]) -> float:
        """Evaluate a single solution."""
        if self.boundary_handler.is_feasible(x):
            self.evaluations += 1
            return self.func(x)
        else:
            return float("inf")

    def evaluate_population(
        self, population: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Evaluate a population of solutions."""
        fitness = np.zeros(population.shape[0])
        budget_left = self.config.budget - self.evaluations

        if budget_left >= population.shape[0]:
            for i in range(population.shape[0]):
                fitness[i] = self.evaluate(population[i])
        else:
            for i in range(budget_left):
                fitness[i] = self.evaluate(population[i])
            fitness[budget_left:] = float("inf")

        return fitness

    def get_logs(self) -> LogDataType:
        """Get all logged data with proper typing."""
        return self.logger.get_logs()

    @abstractmethod
    def optimize(self) -> OptimizationResult[LogDataType]:
        """Run the optimization algorithm."""
        pass
