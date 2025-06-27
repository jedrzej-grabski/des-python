from typing import TYPE_CHECKING, Callable, Type, Union, overload, Literal
import numpy as np
from numpy.typing import NDArray

from src.core.base_optimizer import BaseOptimizer, OptimizationResult

from src.core.config_base import BaseConfig
from src.logging.base_logger import BaseLogData
from src.utils.boundary_handlers import BoundaryHandler, BoundaryHandlerType

# Import specific types for overloads
from src.des.des_optimizer import DESOptimizer
from src.des.config import DESConfig
from src.logging.des_logger import DESLogData


class AlgorithmFactory:
    """Factory for creating optimization algorithm instances."""

    _algorithms: dict[str, Type[BaseOptimizer]] = {}
    _configs: dict[str, Type[BaseConfig]] = {}

    @classmethod
    def register_algorithm(
        cls,
        name: str,
        optimizer_class: Type[BaseOptimizer],
        config_class: Type[BaseConfig],
    ) -> None:
        """Register a new optimization algorithm."""
        cls._algorithms[name] = optimizer_class
        cls._configs[name] = config_class

    # Overloaded signatures for specific algorithms
    @overload
    @classmethod
    def create_optimizer(
        cls,
        algorithm_name: Literal["DES"],
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: DESConfig | None = None,
        boundary_handler: BoundaryHandler | None = None,
        boundary_strategy: BoundaryHandlerType | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        **kwargs,
    ) -> DESOptimizer: ...

    # Generic fallback
    @overload
    @classmethod
    def create_optimizer(
        cls,
        algorithm_name: str,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: BaseConfig | None = None,
        boundary_handler: BoundaryHandler | None = None,
        boundary_strategy: BoundaryHandlerType | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        **kwargs,
    ) -> BaseOptimizer: ...

    @classmethod
    def create_optimizer(
        cls,
        algorithm_name: str,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: BaseConfig | None = None,
        boundary_handler: BoundaryHandler | None = None,
        boundary_strategy: BoundaryHandlerType | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        **kwargs,
    ) -> BaseOptimizer:
        """Create an optimizer instance with proper typing."""
        if algorithm_name not in cls._algorithms:
            available = ", ".join(cls._algorithms.keys())
            raise ValueError(
                f"Unknown algorithm '{algorithm_name}'. Available: {available}"
            )

        optimizer_class = cls._algorithms[algorithm_name]

        # Create default config if none provided
        if config is None:
            config_class = cls._configs[algorithm_name]
            config = config_class(dimensions=len(initial_point))

        return optimizer_class(
            func=func,
            initial_point=initial_point,
            config=config,
            boundary_handler=boundary_handler,
            boundary_strategy=boundary_strategy,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            **kwargs,
        )

    @classmethod
    def get_available_algorithms(cls) -> list[str]:
        """Get list of available algorithm names."""
        return list(cls._algorithms.keys())

    @classmethod
    def create_config(
        cls, algorithm_name: str, dimensions: int, **kwargs
    ) -> BaseConfig:
        """Create a configuration object for the specified algorithm."""
        if algorithm_name not in cls._configs:
            available = ", ".join(cls._configs.keys())
            raise ValueError(
                f"Unknown algorithm '{algorithm_name}'. Available: {available}"
            )

        config_class = cls._configs[algorithm_name]
        return config_class(dimensions=dimensions, **kwargs)
