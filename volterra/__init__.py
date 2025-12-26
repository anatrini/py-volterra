"""
Volterra series implementation for nonlinear audio processing.

This package provides tools for computing and applying Volterra kernels
for harmonic distortion modeling and nonlinear system identification.
"""

from volterra.kernels import VolterraKernel2, ArrayF
from volterra.engines import (
    Volterra2Engine,
    DirectNumpyEngine,
    LowRankEngine,
)
from volterra.processor import VolterraProcessor2
from volterra.sweep import (
    exponential_sweep,
    inverse_filter,
    extract_harmonic_irs,
)

__version__ = "0.1.0"

__all__ = [
    "VolterraKernel2",
    "ArrayF",
    "Volterra2Engine",
    "DirectNumpyEngine",
    "LowRankEngine",
    "VolterraProcessor2",
    "exponential_sweep",
    "inverse_filter",
    "extract_harmonic_irs",
]
