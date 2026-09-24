"""SAGER: Selective Anchor-Guided Emotion Recognition."""

from .backbone import SAGERInputs
from .sager import SAGERConfig, SAGERModel, compute_sager_objective

__all__ = [
    "SAGERConfig",
    "SAGERInputs",
    "SAGERModel",
    "compute_sager_objective",
]
__version__ = "3.0.0"
