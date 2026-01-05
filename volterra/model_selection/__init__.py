"""
Model selection for Volterra system identification.

Provides automatic selection between Memory Polynomial (MP),
Generalized Memory Polynomial (GMP), and full TT-Volterra models.
"""

from volterra.model_selection.selector import (
    ModelSelectionConfig,
    ModelSelector,
)

__all__ = [
    "ModelSelectionConfig",
    "ModelSelector",
]
