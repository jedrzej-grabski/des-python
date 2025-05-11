from typing import final, override
import numpy as np
from numpy.typing import NDArray
from enum import Enum


class BoundaryHandlerType(Enum):
    BOUNCE_BACK = "bounce_back"
    CLAMP = "clamp"


class BoundaryHandler:
    """
    Base class for boundary constraint handling strategies
    """

    def __init__(
        self, lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
    ) -> None:
        """
        Initialize boundary handler with domain bounds.

        Args:
            lower_bounds: Lower bounds for each dimension
            upper_bounds: Upper bounds for each dimension
        """
        self.lower_bounds: NDArray[np.float64] = lower_bounds
        self.upper_bounds: NDArray[np.float64] = upper_bounds

    def repair(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Repair a solution that may violate boundaries.

        Args:
            x: Solution vector to repair

        Returns:
            Repaired solution that satisfies boundary constraints
        """
        raise NotImplementedError("Subclasses must implement the repair method")

    def is_feasible(self, x: NDArray[np.float64]) -> bool:
        """
        Check if a solution is feasible (within bounds).

        Args:
            x: Solution vector to check

        Returns:
            True if solution is within bounds, False otherwise
        """
        return bool(np.all(x >= self.lower_bounds)) and bool(
            np.all(x <= self.upper_bounds)
        )


class BounceBackBoundaryHandler(BoundaryHandler):
    """
    Bounce back boundary handling strategy.
    When a solution violates a boundary constraint, it "bounces back"
    from the boundary into the feasible region.
    """

    @override
    def repair(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Repair solution using bounce back strategy.

        Args:
            x: Solution vector to repair

        Returns:
            Repaired solution
        """
        if self.is_feasible(x):
            return x

        x_repaired = x.copy()

        # Fix lower bound violations
        lower_violations = x < self.lower_bounds
        if np.any(lower_violations):
            indices = np.where(lower_violations)[0]
            for i in indices:
                x_repaired[i] = self.lower_bounds[i] + (self.lower_bounds[i] - x[i]) % (
                    self.upper_bounds[i] - self.lower_bounds[i]
                )

        # Fix upper bound violations
        upper_violations = x > self.upper_bounds
        if np.any(upper_violations):
            indices = np.where(upper_violations)[0]
            for i in indices:
                x_repaired[i] = self.upper_bounds[i] - (x[i] - self.upper_bounds[i]) % (
                    self.upper_bounds[i] - self.lower_bounds[i]
                )

        # Handle any NaN or Inf values
        x_repaired = self._remove_inf_nan(x_repaired)

        # Recursively repair if still infeasible
        if not self.is_feasible(x_repaired):
            return self.repair(x_repaired)

        return x_repaired

    def _remove_inf_nan(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Replace any NaN or Inf values with a large finite value.

        Args:
            x: Array that may contain NaN or Inf values

        Returns:
            Array with NaN and Inf values replaced
        """
        result = x.copy()
        result[np.isnan(result)] = np.finfo(float).max
        result[np.isinf(result)] = np.finfo(float).max
        return result


class ClampBoundaryHandler(BoundaryHandler):
    """
    Clamp boundary handling strategy.
    When a solution violates a boundary constraint, it is clamped to the boundary.
    """

    @override
    def repair(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Repair solution by clamping to boundaries.

        Args:
            x: Solution vector to repair

        Returns:
            Repaired solution
        """
        x_repaired = np.clip(x, self.lower_bounds, self.upper_bounds)
        return self._remove_inf_nan(x_repaired)

    def _remove_inf_nan(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Replace any NaN or Inf values with a large finite value.

        Args:
            x: Array that may contain NaN or Inf values

        Returns:
            Array with NaN and Inf values replaced
        """
        result = x.copy()
        result[np.isnan(result)] = np.finfo(float).max
        result[np.isinf(result)] = np.finfo(float).max
        return result


def create_boundary_handler(
    strategy: BoundaryHandlerType,
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
) -> BoundaryHandler:
    """
    Factory function to create a boundary handler based on the strategy name.

    Args:
        strategy: Type of boundary handling strategy
        lower_bounds: Lower bounds for each dimension
        upper_bounds: Upper bounds for each dimension

    Returns:
        Boundary handler instance

    Raises:
        ValueError: If the strategy is not recognized
    """
    if strategy is BoundaryHandlerType.BOUNCE_BACK:
        return BounceBackBoundaryHandler(lower_bounds, upper_bounds)
    elif strategy is BoundaryHandlerType.CLAMP:
        return ClampBoundaryHandler(lower_bounds, upper_bounds)
