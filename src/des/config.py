from dataclasses import dataclass, field
import numpy as np
import math
from numpy.typing import NDArray


def default_population_size(obj: "DESConfig") -> int:
    """Default population size based on dimensions."""
    return 4 * obj.dimensions


def default_budget(obj: "DESConfig") -> int:
    """Default budget based on dimensions."""
    return 10000 * obj.dimensions


def default_cp(obj: "DESConfig") -> float:
    """Default evolution path decay factor based on dimensions."""
    return 1 / np.sqrt(obj.dimensions)


def default_history(obj: "DESConfig") -> int:
    """Default history size based on dimensions."""
    return math.ceil(6 + math.ceil(3 * np.sqrt(obj.dimensions)))


def default_mu(obj: "DESConfig") -> int:
    """Default number of parents based on population size."""
    return math.floor(obj.population_size / 2)


def default_weights(obj: "DESConfig") -> NDArray[np.float64]:
    """Default recombination weights based on mu."""
    weights = np.log(obj.mu + 1) - np.log(np.arange(1, obj.mu + 1))
    return weights / np.sum(weights)


def default_ccum(obj: "DESConfig") -> float:
    """Default cumulation factor based on mu."""
    return obj.mu / (obj.mu + 2)


def default_pathratio(obj: "DESConfig") -> float:
    """Default path ratio based on path length."""
    return np.sqrt(obj.pathLength)


def compute_maxit(budget: int, population_size: int) -> int:
    """Compute maximum iterations based on budget and population size."""
    return math.floor(budget / (population_size + 1))


def default_ft_scale(obj: "DESConfig") -> float:
    """Default Ft scaling factor."""
    N = obj.dimensions
    mueff = obj.mueff
    return ((mueff + 2) / (N + mueff + 3)) / (
        1
        + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1)
        + (mueff + 2) / (N + mueff + 3)
    )


@dataclass
class DESConfig:
    """
    Configuration for the DES optimizer.

    This class provides a structured way to configure the Differential Evolution
    Strategy (DES) optimizer.
    """

    dimensions: int
    """Number of dimensions in the problem"""

    Ft: float = 1.0
    """Scaling factor of difference vectors"""

    initFt: float = 1.0
    """Initial scaling factor"""

    pathLength: int = 6
    """Size of evolution path"""

    c_Ft: float = 0
    """Control parameter for Ft adaptation"""

    Lamarckism: bool = False
    """Whether to use Lamarckian evolution"""

    budget: int = field(init=False)

    population_size: int = field(init=False)

    cp: float = field(init=False)
    """Evolution path decay factor"""

    history: int = field(init=False)
    """Size of history window"""

    # Parameters with defaults dependent on other parameters
    mu: int = field(init=False)
    """Number of parents"""

    weights: NDArray[np.float64] = field(init=False)
    """Recombination weights"""

    ccum: float = field(init=False)
    """Evolution path decay factor"""

    pathRatio: float = field(init=False)
    """Path length control reference value"""

    maxit: int = field(init=False)
    """Maximum iterations"""

    # Computed parameters
    mueff: float = field(init=False)
    """Effective selection mass"""

    Ft_scale: float = field(init=False)
    """Scaling factor for Ft"""

    # Diagnostic logging options
    diag_enabled: bool = False
    """Enable all diagnostics"""

    diag_Ft: bool = False
    """Log Ft values"""

    diag_value: bool = False
    """Log population fitness values"""

    diag_mean: bool = False
    """Log mean fitness"""

    diag_meanCords: bool = False
    """Log mean coordinates"""

    diag_pop: bool = False
    """Log populations"""

    diag_bestVal: bool = True
    """Log best fitness"""

    diag_worstVal: bool = False
    """Log worst fitness"""

    diag_eigen: bool = False
    """Log eigenvalues"""

    def __post_init__(self) -> None:
        """Calculate derived parameters that depend on other params"""
        # Set dimension-dependent defaults
        self.budget = default_budget(self)
        self.population_size = default_population_size(self)
        self.cp = default_cp(self)
        self.history = default_history(self)

        # Set defaults dependent on other parameters
        self.mu = default_mu(self)
        self.weights = default_weights(self)
        self.ccum = default_ccum(self)
        self.pathRatio = default_pathratio(self)

        # Calculate mueff from weights
        weights_sum_square = np.sum(self.weights**2)
        self.mueff = np.sum(self.weights) ** 2 / weights_sum_square

        # Calculate Ft_scale
        self.Ft_scale = default_ft_scale(self)

        # Calculate maxit
        self.maxit = compute_maxit(self.budget, self.population_size)

    def enable_all_diagnostics(self):
        """Enable all diagnostic logging options."""
        self.diag_enabled = True
        self.diag_Ft = True
        self.diag_value = True
        self.diag_mean = True
        self.diag_meanCords = True
        self.diag_pop = True
        self.diag_bestVal = True
        self.diag_worstVal = True
        self.diag_eigen = True

    def disable_all_diagnostics(self):
        """Disable all diagnostic logging options."""
        self.diag_enabled = False
        self.diag_Ft = False
        self.diag_value = False
        self.diag_mean = False
        self.diag_meanCords = False
        self.diag_pop = False
        self.diag_bestVal = False
        self.diag_worstVal = False
        self.diag_eigen = False

    def with_convergence_diagnostics(self):
        """Enable only diagnostics needed for convergence plots."""
        self.disable_all_diagnostics()
        self.diag_bestVal = True

    def with_custom_budget(self, budget: int):
        """Set custom evaluation budget."""
        self.budget = budget

    def with_population_size(self, size: int):
        """Set custom population size."""
        self.population_size = size
