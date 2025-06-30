"""
Python Evolutionary Optimization Package
"""

from src.algorithms.choices import AlgorithmChoice
from src.core.algorithm_factory import AlgorithmFactory
from src.core.base_optimizer import BaseOptimizer, OptimizationResult
from src.core.config_base import BaseConfig


# Register algorithms using lazy imports to avoid circular dependencies
def _register_algorithms():
    """Register all available algorithms with the factory."""
    try:
        from src.algorithms.des.des_optimizer import DESOptimizer
        from src.algorithms.des.config import DESConfig

        AlgorithmFactory.register_algorithm(
            AlgorithmChoice.DES, DESOptimizer, DESConfig
        )
    except ImportError:
        pass  # Algorithm not available

    # try:
    #     from src.mfcmaes.mfcmaes_optimizer import MFCMAESOptimizer
    #     from src.mfcmaes.config import MFCMAESConfig
    #     AlgorithmFactory.register_algorithm("MFCMAES", MFCMAESOptimizer, MFCMAESConfig)
    # except ImportError:
    #     pass


_register_algorithms()

__all__ = [
    "AlgorithmFactory",
    "BaseOptimizer",
    "OptimizationResult",
    "BaseConfig",
]
