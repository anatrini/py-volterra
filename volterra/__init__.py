"""
Volterra series implementation for nonlinear audio processing.

This package provides tools for computing and applying Volterra kernels
for harmonic distortion modeling and nonlinear system identification.

Features:
---------
- 2nd-order Volterra with full matrix support and low-rank approximation
- Diagonal kernels up to 5th order for real-time performance
- Multi-tone kernel estimation for system identification
- Optimized engines (NumPy and Numba) for efficient computation
- Swept-sine method for 2nd-order kernel extraction

Typical usage:
--------------
    from volterra import VolterraKernelFull, VolterraProcessorFull

    # Create a tube-saturation kernel
    kernel = VolterraKernelFull.from_polynomial_coeffs(
        N=512, a1=0.9, a2=0.12, a3=0.03, a5=0.01
    )

    # Process audio
    proc = VolterraProcessorFull(kernel, sample_rate=48000)
    output = proc.process(input_audio)
"""

# Legacy 2nd-order support (Phase 1)
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

# Full-order support (Phase 2) - orders 1-5
from volterra.kernels_full import VolterraKernelFull
from volterra.engines_diagonal import (
    VolterraFullEngine,
    DiagonalNumpyEngine,
    DiagonalNumbaEngine,
    NUMBA_AVAILABLE,
)
from volterra.processor_full import VolterraProcessorFull
from volterra.estimation import (
    MultiToneConfig,
    MultiToneEstimator,
)

__version__ = "0.2.0"

__all__ = [
    # Legacy 2nd-order API
    "VolterraKernel2",
    "ArrayF",
    "Volterra2Engine",
    "DirectNumpyEngine",
    "LowRankEngine",
    "VolterraProcessor2",
    "exponential_sweep",
    "inverse_filter",
    "extract_harmonic_irs",
    # Full-order API (1-5)
    "VolterraKernelFull",
    "VolterraFullEngine",
    "DiagonalNumpyEngine",
    "DiagonalNumbaEngine",
    "NUMBA_AVAILABLE",
    "VolterraProcessorFull",
    "MultiToneConfig",
    "MultiToneEstimator",
]
