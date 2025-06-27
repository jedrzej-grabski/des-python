"""Algorithm registration module."""

from src.core.algorithm_factory import AlgorithmFactory


def register_all_algorithms():
    """Register all available algorithms."""
    # Register DES
    try:
        from src.des.des_optimizer import DESOptimizer
        from src.des.config import DESConfig

        AlgorithmFactory.register_algorithm("DES", DESOptimizer, DESConfig)
    except ImportError as e:
        print(f"Warning: Could not register DES algorithm: {e}")

    # Add other algorithms here
    # try:
    #     from src.mfcmaes.mfcmaes_optimizer import MFCMAESOptimizer
    #     from src.mfcmaes.config import MFCMAESConfig
    #     AlgorithmFactory.register_algorithm("MFCMAES", MFCMAESOptimizer, MFCMAESConfig)
    # except ImportError as e:
    #     print(f"Warning: Could not register MFCMAES algorithm: {e}")


# Auto-register when module is imported
register_all_algorithms()
