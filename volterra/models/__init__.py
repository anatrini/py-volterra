"""
Volterra system identification models.

This module provides high-level APIs for Volterra system identification:
- TT-Volterra: Tensor-Train based MIMO identification
- Memory polynomial: Diagonal-only Volterra (Hammerstein models)
"""

from volterra.models.tt_volterra import (
    TTVolterraIdentifier,
    TTVolterraConfig,
)

__all__ = [
    "TTVolterraIdentifier",
    "TTVolterraConfig",
]
