"""
Volterra series implementation for nonlinear audio processing.

This package provides tools for computing and applying Volterra kernels
up to 5th order for harmonic distortion modeling.

Features:
---------
- MIMO Tensor-Train (TT) based Volterra identification supporting multiple inputs/outputs
- Diagonal kernels up to 10th order for real-time performance
- Optimized engines (NumPy, Numba, FFT, Vectorized) for efficient computation
- Block-based streaming processor with correct overlap-add
- Mathematical correctness verified with comprehensive test suite

Typical usage:
--------------
    from volterra import VolterraKernelFull, VolterraProcessorFull

    # Create a tube-saturation kernel
    kernel = VolterraKernelFull.from_polynomial_coeffs(
        N=512, a1=0.9, a2=0.12, a3=0.03, a5=0.01
    )

    # Process audio in blocks
    proc = VolterraProcessorFull(kernel, sample_rate=48000)
    output = proc.process(input_audio, block_size=512)

System identification uses Tensor-Train decomposition for MIMO Volterra systems.
See TTVolterraIdentifier for details.
"""

from volterra.kernels_full import VolterraKernelFull, ArrayF
from volterra.engines_diagonal import (
    VolterraFullEngine,
    DiagonalNumpyEngine,
    DiagonalNumbaEngine,
    NUMBA_AVAILABLE,
)
from volterra.processor_full import VolterraProcessorFull

# System identification (TT-based MIMO Volterra)
from volterra.models import (
    TTVolterraIdentifier,
    TTVolterraConfig,
)

# Optimized engines (optional, requires Numba)
try:
    from volterra.engines_optimized import (
        OptimizedDiagonalEngine,
        OptimizedNumbaEngine,
    )
    __all_optional__ = [
        "OptimizedDiagonalEngine",
        "OptimizedNumbaEngine",
    ]
except ImportError:
    __all_optional__ = []

# FFT-optimized engines (automatic selection in processor)
try:
    from volterra.engines_fft import (
        FFTOptimizedEngine,
        FFTOptimizedNumbaEngine,
    )
    __all_optional__.extend([
        "FFTOptimizedEngine",
        "FFTOptimizedNumbaEngine",
    ])
except ImportError:
    pass

# Vectorized engines (NO Python loops)
try:
    from volterra.engines_vectorized import VectorizedEngine
    __all_optional__.append("VectorizedEngine")
except ImportError:
    pass

__version__ = "0.6.0"

__all__ = [
    # Core types
    "VolterraKernelFull",
    "ArrayF",
    # Engines
    "VolterraFullEngine",
    "DiagonalNumpyEngine",
    "DiagonalNumbaEngine",
    "NUMBA_AVAILABLE",
    # Processor
    "VolterraProcessorFull",
    # System identification
    "TTVolterraIdentifier",
    "TTVolterraConfig",
] + __all_optional__
