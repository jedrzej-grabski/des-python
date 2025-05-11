"""
DES Algorithm - Differential Evolution with Success-History Based Parameter Adaptation
"""

from des.des_optimizer import DESOptimizer
from des.utils.boundary_handlers import (
    BoundaryHandler,
    BounceBackBoundaryHandler,
    ClampBoundaryHandler,
)

__version__ = "0.1.0"
__all__ = [
    "DESOptimizer",
    "BoundaryHandler",
    "BounceBackBoundaryHandler",
    "ClampBoundaryHandler",
]
