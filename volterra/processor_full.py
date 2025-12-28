"""
Extended Volterra processor supporting orders 1-5 with optimal performance.

This processor handles streaming audio with proper state management and
achieves real-time performance for diagonal kernels up to 5th order.
"""

from dataclasses import dataclass
import numpy as np
from scipy.signal import lfilter
from volterra.kernels_full import VolterraKernelFull, ArrayF
from volterra.engines_diagonal import (
    VolterraFullEngine,
    DiagonalNumpyEngine,
    DiagonalNumbaEngine,
    NUMBA_AVAILABLE
)

# Try to import FFT-optimized engines
try:
    from volterra.engines_fft import (
        FFTOptimizedEngine,
        FFTOptimizedNumbaEngine
    )
    FFT_ENGINES_AVAILABLE = True
except ImportError:
    FFT_ENGINES_AVAILABLE = False


@dataclass
class VolterraProcessorFull:
    """
    Streaming Volterra processor for orders 1-5.

    Handles block-based processing with correct history management
    for real-time and offline applications.

    Performance targets (N=512, block_size=512):
    - h1 only: ~0.1 ms/block
    - h1+h2: ~0.3 ms/block
    - h1+h2+h3: ~0.5 ms/block
    - h1+h2+h3+h5: ~0.8 ms/block (Numba)
    - h1+h2+h3+h4+h5: ~1.0 ms/block (Numba)

    Example:
        >>> kernel = VolterraKernelFull.from_polynomial_coeffs(
        ...     N=512, a1=0.9, a2=0.1, a3=0.02, a5=0.005
        ... )
        >>> proc = VolterraProcessorFull(kernel, sample_rate=48000)
        >>> y = proc.process(x)
    """

    kernel: VolterraKernelFull
    engine: VolterraFullEngine = None
    sample_rate: int = 48000
    use_numba: bool = True

    def __post_init__(self):
        """Initialize processor and select optimal engine."""
        # Auto-select engine if not specified
        if self.engine is None:
            N = self.kernel.N

            # Optimal selection based on kernel length
            # FFT crossover: ~128 samples
            if FFT_ENGINES_AVAILABLE and N >= 128:
                # Use FFT-optimized engine for long kernels
                if self.use_numba and NUMBA_AVAILABLE:
                    print(f"Using FFT+Numba hybrid engine (N={N}, max_order={self.kernel.max_order})")
                    object.__setattr__(
                        self,
                        'engine',
                        FFTOptimizedNumbaEngine(
                            self.kernel,
                            fft_threshold=128,
                            max_block_size=4096
                        )
                    )
                else:
                    print(f"Using FFT-optimized engine (N={N}, max_order={self.kernel.max_order})")
                    object.__setattr__(
                        self,
                        'engine',
                        FFTOptimizedEngine(
                            self.kernel,
                            fft_threshold=128,
                            max_block_size=4096
                        )
                    )
            else:
                # Use time-domain engine for short kernels
                if self.use_numba and NUMBA_AVAILABLE:
                    print(f"Using Numba time-domain engine (N={N}, max_order={self.kernel.max_order})")
                    object.__setattr__(self, 'engine', DiagonalNumbaEngine())
                else:
                    if self.use_numba and not NUMBA_AVAILABLE:
                        print("Warning: Numba requested but not available, using NumPy")
                    print(f"Using NumPy time-domain engine (N={N}, max_order={self.kernel.max_order})")
                    object.__setattr__(self, 'engine', DiagonalNumpyEngine())

        # Estimate memory
        mem_kb = self.kernel.estimate_memory_bytes() / 1024
        print(f"Kernel memory footprint: {mem_kb:.1f} KB")

        self.reset()

    def reset(self):
        """Reset processor state."""
        N = self.kernel.N
        object.__setattr__(self, '_x_history', np.zeros(N - 1, dtype=np.float64))

        # State for linear convolution (scipy.signal.lfilter)
        object.__setattr__(self, '_zi1', np.zeros(N - 1, dtype=np.float64))

    def process_block(self, x_block: ArrayF) -> ArrayF:
        """
        Process one block through all active Volterra orders.

        Args:
            x_block: Input block of samples

        Returns:
            Output block (same length as input)
        """
        x_block = np.asarray(x_block, dtype=np.float64)

        # Use engine for all nonlinear orders
        y = self.engine.process_block(x_block, self._x_history, self.kernel)

        # Update history for next block
        N = self.kernel.N
        B = len(x_block)

        if B >= N - 1:
            # Block larger than kernel: use last N-1 samples
            object.__setattr__(self, '_x_history', x_block[-(N-1):].copy())
        else:
            # Block smaller than kernel: concatenate and trim
            new_history = np.concatenate([self._x_history, x_block])
            object.__setattr__(self, '_x_history', new_history[-(N-1):].copy())

        return y

    def process(self, x: ArrayF, block_size: int = 512) -> ArrayF:
        """
        Process complete signal offline.

        Args:
            x: Input signal
            block_size: Processing block size (larger = more efficient)

        Returns:
            Processed signal
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.empty_like(x)
        self.reset()

        for i in range(0, len(x), block_size):
            xb = x[i:i+block_size]
            yb = self.process_block(xb)
            y[i:i+len(xb)] = yb

        return y

    def get_info(self) -> dict:
        """Get processor configuration info."""
        return {
            'kernel_length': self.kernel.N,
            'max_order': self.kernel.max_order,
            'sample_rate': self.sample_rate,
            'engine': self.engine.__class__.__name__,
            'memory_kb': self.kernel.estimate_memory_bytes() / 1024,
            'active_kernels': {
                'h1': True,
                'h2': self.kernel.h2 is not None,
                'h3': self.kernel.h3_diagonal is not None,
                'h4': self.kernel.h4_diagonal is not None,
                'h5': self.kernel.h5_diagonal is not None,
            }
        }
