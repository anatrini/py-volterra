"""
Volterra system identification models.

This module provides high-level APIs for Volterra system identification:
- TTVolterraIdentifier: Diagonal TT-Volterra (memory polynomial)
- TTVolterraMIMO: Full TT-Volterra with arbitrary ranks
- GeneralizedMemoryPolynomial: GMP with selective cross-terms
- Memory polynomial: Diagonal-only Volterra (Hammerstein models)
"""

from volterra.models.tt_volterra import (
    TTVolterraIdentifier,
    TTVolterraConfig,
)

from volterra.models.tt_volterra_full import (
    TTVolterraMIMO,
    TTVolterraFullConfig,
)

from volterra.models.gmp import (
    GeneralizedMemoryPolynomial,
    GMPConfig,
)

__all__ = [
    "TTVolterraIdentifier",
    "TTVolterraConfig",
    "TTVolterraMIMO",
    "TTVolterraFullConfig",
    "GeneralizedMemoryPolynomial",
    "GMPConfig",
]
