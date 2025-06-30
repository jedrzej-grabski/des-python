from dataclasses import dataclass
import numpy as np

from core.config_base import BaseConfig


def default_mfcmaes_population_size(obj: "MFCMAESConfig") -> int:
    """Default population size for MFCMAES based on dimensions."""
    return 4 + int(3 * np.log(obj.dimensions))


def default_mfcmaes_budget(obj: "MFCMAESConfig") -> int:
    """Default budget for MFCMAES based on dimensions."""
    return 1000 * obj.dimensions


@dataclass
class MFCMAESConfig(BaseConfig):
    pass
