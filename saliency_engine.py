"""
OptiSight Saliency Engine (Compatibility Facade)
Re-exports all engine symbols from the modular `core.saliency` package.
"""

from core.saliency import (
    SaliencyEngine,
    Quadrant,
    QUADRANT_LABELS_TR,
    QUADRANT_LABELS_EN,
    _cached_center_bias,
)

__all__ = [
    "SaliencyEngine",
    "Quadrant",
    "QUADRANT_LABELS_TR",
    "QUADRANT_LABELS_EN",
    "_cached_center_bias",
]
