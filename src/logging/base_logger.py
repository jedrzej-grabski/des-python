from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, TypeVar, Generic
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

# Create a generic type variable for log data
LogDataType = TypeVar("LogDataType", bound="BaseLogData")


class LoggerProtocol(Protocol):
    """Protocol for logger objects."""

    def log_iteration(self, **kwargs) -> None: ...
    def get_logs(self) -> Any: ...
    def clear_logs(self) -> None: ...


@dataclass
class BaseLogData:
    """Base container for diagnostic log data shared across algorithms."""

    # Common data across all algorithms
    iteration: List[int] = field(default_factory=list)
    evaluations: List[int] = field(default_factory=list)
    best_fitness: List[float] = field(default_factory=list)
    worst_fitness: List[float] = field(default_factory=list)
    mean_fitness: List[float] = field(default_factory=list)
    std_fitness: List[float] = field(default_factory=list)
    population: List[NDArray[np.float64]] = field(default_factory=list)
    best_solution: List[NDArray[np.float64]] = field(default_factory=list)

    # Optional common data
    eigenvalues: List[NDArray[np.float64]] = field(default_factory=list)
    condition_number: List[float] = field(default_factory=list)

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

    def to_dict_common(self) -> Dict[str, List[Any]]:
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

    def to_dict(self) -> Dict[str, List[Any]]:
        """Convert all log data to dictionary format - to be overridden in subclasses."""
        return self.to_dict_common()


class BaseLogger(ABC, Generic[LogDataType]):
    """Base class for algorithm-specific loggers."""

    def __init__(self, config, algorithm_name: str = "Unknown"):
        self.config = config
        self.algorithm_name = algorithm_name
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

    def get_logs_dict(self) -> Dict[str, Any]:
        """Get all logged data as dictionary."""
        return self.logs.to_dict()

    def clear_logs(self) -> None:
        """Clear all logged data."""
        self.logs.clear()

    def get_algorithm_name(self) -> str:
        """Get the algorithm name."""
        return self.algorithm_name
