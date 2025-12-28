"""
Volterra series implementation for nonlinear audio processing.

This package provides tools for computing and applying Volterra kernels
up to 5th order for harmonic distortion modeling and nonlinear system identification.

Features:
---------
- Diagonal kernels up to 5th order for real-time performance
- Multi-tone kernel estimation for system identification
- Optimized engines (NumPy and Numba) for efficient computation
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

System identification:
----------------------
    from volterra import MultiToneConfig, MultiToneEstimator

    # Generate excitation signal
    config = MultiToneConfig(num_tones=100, max_order=5)
    estimator = MultiToneEstimator(config)
    excitation, freqs = estimator.generate_excitation()

    # Record system response and extract kernels
    kernel = estimator.estimate_kernel(excitation, response, freqs)
"""

from volterra.kernels_full import VolterraKernelFull, ArrayF
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

__version__ = "0.4.0"

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
    # Estimation
    "MultiToneConfig",
    "MultiToneEstimator",
] + __all_optional__
