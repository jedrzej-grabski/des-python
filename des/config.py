from dataclasses import dataclass
import numpy as np
import math


@dataclass
class DESConfig:
    """
    Configuration for the DES optimizer.

    This class provides a structured way to configure the Differential Evolution
    with Success-History Based Parameter Adaptation (DES) optimizer.
    """

    # Algorithm parameters
    Ft: float = 1.0
    """Scaling factor of difference vectors"""

    initFt: float = 1.0
    """Initial scaling factor"""

    stopfitness: float = -float("inf")
    """Target fitness value to stop optimization when reached"""

    budget: int | None = None
    """Maximum function evaluations (default: 10000 * dimensions)"""

    population_size: int | None = None  # Renamed from lambda
    """Population size (default: 4 * dimensions)"""

    minlambda: int | None = None
    """Minimum population size (default: 4 * dimensions)"""

    mu: int | None = None
    """Number of parents (default: population_size/2)"""

    weights: np.ndarray | None = None
    """Recombination weights (computed automatically if None)"""

    ccum: float | None = None
    """Evolution path decay factor (default: mu/(mu+2))"""

    pathLength: int = 6
    """Size of evolution path"""

    cp: float | None = None
    """Evolution path decay factor (default: 1/sqrt(dimensions))"""

    maxit: int | None = None
    """Maximum iterations (derived from budget if None)"""

    c_Ft: float = 0
    """Control parameter for Ft adaptation"""

    pathRatio: float | None = None
    """Path length control reference value (default: sqrt(pathLength))"""

    history: int | None = None
    """Size of history window (default: 6 + ceil(3 * sqrt(dimensions)))"""

    Ft_scale: float | None = None
    """Scaling factor for Ft (computed automatically if None)"""

    tol: float = 1e-12
    """Convergence tolerance"""

    Lamarckism: bool = False
    """Whether to use Lamarckian evolution"""

    # Derived parameters (computed during initialization)
    mueff: float | None = None
    """Effective selection mass"""

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

    def prepare(self, dimensions: int) -> None:
        """
        Prepare configuration with dimension-dependent defaults.

        Args:
            dimensions: Number of dimensions in the problem
        """
        N = dimensions

        # Set dimension-dependent defaults
        if self.budget is None:
            self.budget = 10000 * N

        if self.population_size is None:
            self.population_size = 4 * N

        if self.minlambda is None:
            self.minlambda = 4 * N

        if self.cp is None:
            self.cp = 1 / np.sqrt(N)

        if self.history is None:
            self.history = math.ceil(6 + math.ceil(3 * np.sqrt(N)))

        # Compute derived parameters
        if self.mu is None:
            self.mu = math.floor(self.population_size / 2)

        if self.weights is None:
            weights = np.log(self.mu + 1) - np.log(np.arange(1, self.mu + 1))
            self.weights = weights / np.sum(weights)

        weights_sum_square = np.sum(self.weights**2)
        self.mueff = np.sum(self.weights) ** 2 / weights_sum_square

        if self.ccum is None:
            self.ccum = self.mu / (self.mu + 2)

        if self.pathRatio is None:
            self.pathRatio = np.sqrt(self.pathLength)

        if self.maxit is None:
            self.maxit = math.floor(self.budget / (self.population_size + 1))

        if self.Ft_scale is None:
            self.Ft_scale = ((self.mueff + 2) / (N + self.mueff + 3)) / (
                1
                + 2 * max(0, np.sqrt((self.mueff - 1) / (N + 1)) - 1)
                + (self.mueff + 2) / (N + self.mueff + 3)
            )

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
        return self

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
        return self

    def with_convergence_diagnostics(self):
        """Enable only diagnostics needed for convergence plots."""
        self.disable_all_diagnostics()
        self.diag_bestVal = True
        return self

    def with_custom_budget(self, budget: int):
        """Set custom evaluation budget."""
        self.budget = budget
        return self

    def with_population_size(self, size: int):
        """Set custom population size."""
        self.population_size = size
        return self

    def with_tolerance(self, tolerance: float):
        """Set convergence tolerance."""
        self.tol = tolerance
