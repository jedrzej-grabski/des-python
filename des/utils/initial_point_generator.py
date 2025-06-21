from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


class InitialPointGeneratorType(Enum):
    """
    Enum for different types of initial point generators.
    """

    NORMAL = "normal"
    UNIFORM = "uniform"
    CUSTOM = "custom"
    ZERO = "zero"

    def __str__(self):
        return self.value


@dataclass
class InitialPointGenerator:
    """
    Base class for generating initial points for optimization algorithms.
    """

    dimensions: int
    lower_bounds: int | float | NDArray[np.float64]
    upper_bounds: int | float | NDArray[np.float64]
    initial_point: NDArray[np.float64] | None = None
    custom_std: NDArray[np.float64] | None = None
    strategy: InitialPointGeneratorType = InitialPointGeneratorType.NORMAL

    def __post_init__(self):
        """Handle bound conversion after initialization."""
        if isinstance(self.lower_bounds, (int, float)):
            self.lower_bounds = np.full(self.dimensions, self.lower_bounds)
        else:
            self.lower_bounds = np.array(self.lower_bounds, dtype=float)

        if isinstance(self.upper_bounds, (int, float)):
            self.upper_bounds = np.full(self.dimensions, self.upper_bounds)
        else:
            self.upper_bounds = np.array(self.upper_bounds, dtype=float)

    def generate(self) -> NDArray[np.float64]:
        """
        Generate an initial point.

        Returns:
            List of floats representing the initial point
        """
        match self.strategy:
            case InitialPointGeneratorType.NORMAL:
                return self._generate_normal()
            case InitialPointGeneratorType.UNIFORM:
                return self._generate_uniform()
            case InitialPointGeneratorType.CUSTOM:
                return self._generate_custom()
            case InitialPointGeneratorType.ZERO:
                return self._generate_zero()
            case _:
                raise ValueError(f"Unknown strategy: {self.strategy}")

    def _generate_normal(self) -> NDArray[np.float64]:
        """
        Generate an initial point using a normal distribution.

        This method uses the mean and standard deviation based on the bounds
        and optionally custom mean and standard deviation if provided.

        If initial_point is probided, it will be used as the mean of t he normal distribution.

        Returns:
            Array of floats sampled from a normal distribution
        """
        assert isinstance(self.lower_bounds, np.ndarray) and isinstance(
            self.upper_bounds, np.ndarray
        ), "Bounds must be numpy arrays"

        if (
            self.initial_point is not None
            and len(self.initial_point) == self.dimensions
        ):
            mean = np.array(self.initial_point, dtype=np.float64)
        else:
            mean = np.array(
                [
                    (lower + upper) / 2
                    for lower, upper in zip(self.lower_bounds, self.upper_bounds)
                ]
            )

        if self.custom_std and len(self.custom_std) == self.dimensions:
            std = np.array(self.custom_std, dtype=np.float64)
        else:
            # 6-sigma
            std = np.array(
                [
                    (upper - lower) / 6
                    for lower, upper in zip(self.lower_bounds, self.upper_bounds)
                ]
            )

        point = np.random.normal(mean, std, size=self.dimensions)
        return np.clip(point, self.lower_bounds, self.upper_bounds)

    def _generate_uniform(self) -> NDArray[np.float64]:
        """
        Generate an initial point using a uniform distribution.

        Returns:
            Array of floats sampled uniformly from the bounds
        """
        return np.random.uniform(
            self.lower_bounds, self.upper_bounds, size=self.dimensions
        )

    def _generate_custom(self) -> NDArray[np.float64]:
        """
        Generate an initial point using a custom predefined point.

        Returns:
            Array of floats from the predefined initial point
        """
        if self.initial_point is None:
            raise ValueError("Custom initial point not provided")

        elif len(self.initial_point) != self.dimensions:
            raise ValueError(
                f"Initial point length {len(self.initial_point)} does not match dimensions {self.dimensions}"
            )
        return np.array(self.initial_point, dtype=np.float64)

    def _generate_zero(self) -> NDArray[np.float64]:
        """
        Generate an initial point with all zeros.

        Returns:
            Array of zeros
        """
        return np.zeros(self.dimensions, dtype=np.float64)
