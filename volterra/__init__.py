"""
Volterra series implementation for nonlinear audio processing.

This package provides tools for computing and applying Volterra kernels
up to 5th order for harmonic distortion modeling, plus system identification
using Tensor-Train decomposition.

Features:
---------
- MIMO Tensor-Train (TT) based Volterra identification
- Diagonal kernels up to 10th order for real-time performance
- Optimized engines (NumPy, Numba, FFT, Vectorized)
- Block-based streaming processor with overlap-add
- Composable pipelines (nonlinear → RIR acoustic chains)
- Comprehensive test suite (165+ tests, 72% coverage)

Workflow:
---------

**1. System Identification** (fit model from data):

    from volterra import TTVolterraIdentifier

    # Identify nonlinear system from input-output measurements
    identifier = TTVolterraIdentifier(
        memory_length=20,
        order=3,
        ranks=[1, 5, 3, 1]
    )
    identifier.fit(x_measured, y_measured)

    # Use for prediction
    y_pred = identifier.predict(x_new)

**2. Real-time Processing** (apply known kernel):

    from volterra import VolterraKernelFull, VolterraProcessorFull

    # Create kernel from polynomial coefficients
    kernel = VolterraKernelFull.from_polynomial_coeffs(
        N=512, a1=0.9, a2=0.12, a3=0.03
    )

    # Process audio in blocks
    proc = VolterraProcessorFull(kernel, sample_rate=48000)
    output = proc.process(input_audio, block_size=512)

**3. Acoustic Chain Composition** (nonlinear → room):

    from volterra import TTVolterraIdentifier, NonlinearThenRIR

    # Stage 1: Identify source nonlinearity
    identifier = TTVolterraIdentifier(memory_length=15, order=2, ranks=[1, 4, 1])
    identifier.fit(x_source, y_source)

    # Stage 2: Compose with room impulse response
    chain = NonlinearThenRIR(
        nonlinear_model=identifier,
        rir=room_impulse_response,
        sample_rate=48000
    )

    # Process through full chain
    y_recorded = chain.process(x_instrument)

See documentation for TTVolterraIdentifier, VolterraProcessorFull, and
NonlinearThenRIR for detailed examples.
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

# Pipelines (composable processing chains)
from volterra.pipelines import (
    NonlinearThenRIR,
    AcousticChainConfig,
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
    # Pipelines
    "NonlinearThenRIR",
    "AcousticChainConfig",
] + __all_optional__
