"""
Volterra system identification models.

This module provides high-level APIs for Volterra system identification:
- TTVolterraIdentifier: Diagonal TT-Volterra (memory polynomial)
- TTVolterraMIMO: Full TT-Volterra with arbitrary ranks
- GeneralizedMemoryPolynomial: GMP with selective cross-terms
- Memory polynomial: Diagonal-only Volterra (Hammerstein models)
"""

from volterra.models.gmp import (
    GeneralizedMemoryPolynomial,
    GMPConfig,
)
from volterra.models.tt_volterra import (
    TTVolterraConfig,
    TTVolterraIdentifier,
)
from volterra.models.tt_volterra_full import (
    TTVolterraFullConfig,
    TTVolterraMIMO,
)

__all__ = [
    "GMPConfig",
    "GeneralizedMemoryPolynomial",
    "TTVolterraConfig",
    "TTVolterraFullConfig",
    "TTVolterraIdentifier",
    "TTVolterraMIMO",
]
