from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar, Generic
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from src.algorithms.choices import AlgorithmChoice

LogDataType = TypeVar("LogDataType", bound="BaseLogData")


class LoggerProtocol(Protocol):
    """Protocol for logger objects."""

    def log_iteration(self, **kwargs) -> None: ...
    def get_logs(self) -> Any: ...
    def clear_logs(self) -> None: ...


@dataclass
class BaseLogData:
    """Base container for diagnostic log data shared across algorithms."""

    iteration: list[int] = field(default_factory=list)
    evaluations: list[int] = field(default_factory=list)
    best_fitness: list[float] = field(default_factory=list)
    worst_fitness: list[float] = field(default_factory=list)
    mean_fitness: list[float] = field(default_factory=list)
    std_fitness: list[float] = field(default_factory=list)
    population: list[NDArray[np.float64]] = field(default_factory=list)
    best_solution: list[NDArray[np.float64]] = field(default_factory=list)

    eigenvalues: list[NDArray[np.float64]] = field(default_factory=list)
    condition_number: list[float] = field(default_factory=list)

    def clear_common(self) -> None:
        """Reset common log data."""
        self.iteration.clear()
        self.evaluations.clear()
        self.best_fitness.clear()
        self.worst_fitness.clear()
        self.mean_fitness.clear()
        self.std_fitness.clear()
        self.population.clear()
        self.best_solution.clear()
        self.eigenvalues.clear()
        self.condition_number.clear()

    def to_dict_common(self) -> dict[str, list[Any]]:
        """Convert common log data to dictionary format."""
        return {
            "iteration": self.iteration,
            "evaluations": self.evaluations,
            "best_fitness": self.best_fitness,
            "worst_fitness": self.worst_fitness,
            "mean_fitness": self.mean_fitness,
            "std_fitness": self.std_fitness,
            "population": self.population,
            "best_solution": self.best_solution,
            "eigenvalues": self.eigenvalues,
            "condition_number": self.condition_number,
        }

    def clear(self) -> None:
        """Reset all log data - to be overridden in subclasses."""
        self.clear_common()

    def to_dict(self) -> dict[str, list[Any]]:
        """Convert all log data to dictionary format - to be overridden in subclasses."""
        return self.to_dict_common()


class BaseLogger(ABC, Generic[LogDataType]):
    """Base class for algorithm-specific loggers."""

    def __init__(
        self, config, algorithm: AlgorithmChoice = AlgorithmChoice.Unknown
    ) -> None:
        self.config = config
        self.algorithm = algorithm
        self.logs: LogDataType = self._create_log_data()

    @abstractmethod
    def _create_log_data(self) -> LogDataType:
        """Create algorithm-specific log data container."""
        pass

    @abstractmethod
    def log_iteration(self, **kwargs) -> None:
        """Log iteration data."""
        pass

    def get_logs(self) -> LogDataType:
        """Get all logged data."""
        return self.logs

    def get_logs_dict(self) -> dict[str, Any]:
        """Get all logged data as dictionary."""
        return self.logs.to_dict()

    def clear_logs(self) -> None:
        """Clear all logged data."""
        self.logs.clear()

    def get_algorithm(self) -> AlgorithmChoice:
        """Get the algorithm name."""
        return self.algorithm
