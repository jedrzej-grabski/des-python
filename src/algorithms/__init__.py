"""Algorithm registration module."""

from src.core.algorithm_factory import AlgorithmFactory
from src.algorithms.choices import AlgorithmChoice


def register_all_algorithms():
    """Register all available algorithms."""
    try:
        from src.algorithms.des.des_optimizer import DESOptimizer
        from src.algorithms.des.config import DESConfig

        AlgorithmFactory.register_algorithm(
            AlgorithmChoice.DES, DESOptimizer, DESConfig
        )
    except ImportError as e:
        print(f"Warning: Could not register DES algorithm: {e}")


register_all_algorithms()

__all__ = [
    "AlgorithmChoice",
]
