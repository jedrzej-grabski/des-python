from dataclasses import dataclass, field
import numpy as np
import math
from numpy.typing import NDArray

from core.config_base import BaseConfig


def default_mfcmaes_population_size(obj: "MFCMAESConfig") -> int:
    """Default population size for MFCMAES based on dimensions."""
    return 4 + int(3 * np.log(obj.dimensions))


def default_mfcmaes_budget(obj: "MFCMAESConfig") -> int:
    """Default budget for MFCMAES based on dimensions."""
    return 1000 * obj.dimensions


@dataclass
class MFCMAESConfig(BaseConfig):
    """
    Configuration for the MFCMAES optimizer.
    Extends BaseConfig with MFCMAES-specific parameters.
    """

    # MFCMAES-specific parameters
    sigma: float = 1.0
    """Initial step size"""

    cc: float = field(init=False)
    """Cumulation constant for C"""

    cs: float = field(init=False)
    """Cumulation constant for sigma"""

    c1: float = field(init=False)
    """Learning rate for rank-1 update"""

    cmu: float = field(init=False)
    """Learning rate for rank-mu update"""

    damps: float = field(init=False)
    """Damping for sigma"""

    # Computed/derived parameters
    mu: int = field(init=False)
    """Number of parents"""

    weights: NDArray[np.float64] = field(init=False)
    """Recombination weights"""

    mueff: float = field(init=False)
    """Effective selection mass"""

    # MFCMAES-specific diagnostic logging
    diag_sigma: bool = False
    """Log sigma values"""

    diag_C: bool = False
    """Log covariance matrix"""

    def __post_init__(self) -> None:
        """Calculate derived parameters that depend on other params"""
        # Set dimension-dependent defaults
        self.budget = default_mfcmaes_budget(self)
        self.population_size = default_mfcmaes_population_size(self)

        # Set MFCMAES-specific defaults
        self.mu = self.population_size // 2

        # Create weights
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)

        # Calculate mueff
        self.mueff = np.sum(self.weights) ** 2 / np.sum(self.weights**2)

        # Set learning rates
        self.cc = (4 + self.mueff / self.dimensions) / (
            self.dimensions + 4 + 2 * self.mueff / self.dimensions
        )
        self.cs = (self.mueff + 2) / (self.dimensions + self.mueff + 5)
        self.c1 = 2 / ((self.dimensions + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2
            * (self.mueff - 2 + 1 / self.mueff)
            / ((self.dimensions + 2) ** 2 + self.mueff),
        )
        self.damps = (
            1
            + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dimensions + 1)) - 1)
            + self.cs
        )

        # Call parent validation
        super().validate()

    def enable_all_diagnostics(self) -> None:
        """Enable all diagnostic logging options including MFCMAES-specific ones."""
        super().enable_all_diagnostics()
        self.diag_sigma = True
        self.diag_C = True
