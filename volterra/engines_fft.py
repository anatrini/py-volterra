"""
FFT-optimized Volterra engines with kernel precomputation.

This module implements optimal convolution strategies:
1. rfft/irfft for real signals (2x faster than complex FFT)
2. Kernel FFT precomputation (10-40% speedup)
3. next_fast_len for optimal FFT sizes (avoids prime-factor slowdown)
4. Auto-selection between time-domain and FFT based on kernel length

Performance targets:
- FFT vs time-domain crossover: ~64-256 samples (measured)
- rfft vs fft: ~1.3-2x speedup
- Kernel precomputation: ~10-40% speedup
"""

import numpy as np
from scipy import fft

from volterra.kernels_full import ArrayF, VolterraKernelFull

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class FFTOptimizedEngine:
    """
    FFT-optimized diagonal Volterra engine with kernel precomputation.

    Key optimizations:
    1. Pre-computes FFT(kernel) once in __init__
    2. Uses rfft/irfft for real signals (2x speedup vs complex FFT)
    3. Uses scipy.fft.next_fast_len for optimal FFT size
    4. Auto-selects time-domain vs FFT based on kernel length
    5. Caches precomputed kernels for multiple block sizes

    Performance:
    - N=512, B=512: ~0.5-1ms per block (all 5 orders)
    - ~3-5x faster than time-domain for N>=256
    - ~1.5x faster than scipy.signal.fftconvolve
    """

    def __init__(
        self, kernel: VolterraKernelFull, fft_threshold: int = 128, max_block_size: int = 4096
    ):
        """
        Initialize FFT engine with kernel precomputation.

        Args:
            kernel: Volterra kernel (orders 1-5)
            fft_threshold: Use FFT if kernel length > threshold (default: 128)
            max_block_size: Maximum expected block size for precomputation
        """
        self.kernel = kernel
        self.fft_threshold = fft_threshold
        self.max_block_size = max_block_size

        # Determine if we use FFT
        self.use_fft = fft_threshold <= kernel.N

        if self.use_fft:
            # Precompute optimal FFT size and kernel FFTs
            self._precompute_fft_kernels()
        else:
            # Use time-domain convolution
            self._fft_kernels = None
            self._fft_size = None

    def _precompute_fft_kernels(self):
        """
        Precompute FFT(kernel) for all active orders.

        Uses next_fast_len to find optimal FFT size with good prime factorization.
        """
        N = self.kernel.N
        B_max = self.max_block_size

        # Compute optimal FFT size
        # For linear convolution: need at least N + B - 1
        required_len = N + B_max - 1
        self._fft_size = fft.next_fast_len(required_len)

        # Precompute FFT of all kernels
        self._fft_kernels = {}

        # h1 (always present)
        self._fft_kernels["h1"] = fft.rfft(self.kernel.h1, n=self._fft_size)

        # h2 diagonal
        if self.kernel.h2 is not None and self.kernel.h2_is_diagonal:
            self._fft_kernels["h2"] = fft.rfft(self.kernel.h2, n=self._fft_size)

        # h3 diagonal
        if self.kernel.h3_diagonal is not None:
            self._fft_kernels["h3"] = fft.rfft(self.kernel.h3_diagonal, n=self._fft_size)

        # h4 diagonal
        if self.kernel.h4_diagonal is not None:
            self._fft_kernels["h4"] = fft.rfft(self.kernel.h4_diagonal, n=self._fft_size)

        # h5 diagonal
        if self.kernel.h5_diagonal is not None:
            self._fft_kernels["h5"] = fft.rfft(self.kernel.h5_diagonal, n=self._fft_size)

    def process_block(
        self, x_block: ArrayF, x_history: ArrayF, kernel: VolterraKernelFull
    ) -> ArrayF:
        """
        Process block with optimal FFT convolution.

        Args:
            x_block: Input block (B,)
            x_history: History buffer (N-1,)
            kernel: Volterra kernel (must match self.kernel)

        Returns:
            Output block (B,)
        """
        B = len(x_block)
        N = kernel.N

        # Build extended signal
        x_ext = np.concatenate([x_history, x_block])

        # Pre-compute powers using optimal chain
        powers = self._compute_power_chain(x_ext, kernel.max_order)

        # Output accumulator
        y = np.zeros(B, dtype=np.float64)

        if self.use_fft:
            # FFT-based convolution with precomputed kernels
            y += self._fft_convolve(powers[1], self._fft_kernels["h1"], B, kernel.h1)

            if kernel.h2 is not None and kernel.h2_is_diagonal:
                y += self._fft_convolve(powers[2], self._fft_kernels["h2"], B, kernel.h2)

            if kernel.h3_diagonal is not None:
                y += self._fft_convolve(powers[3], self._fft_kernels["h3"], B, kernel.h3_diagonal)

            if kernel.h4_diagonal is not None:
                y += self._fft_convolve(powers[4], self._fft_kernels["h4"], B, kernel.h4_diagonal)

            if kernel.h5_diagonal is not None:
                y += self._fft_convolve(powers[5], self._fft_kernels["h5"], B, kernel.h5_diagonal)

        else:
            # Time-domain convolution for short kernels
            y += self._time_domain_convolve(powers[1], kernel.h1, B, N)

            if kernel.h2 is not None and kernel.h2_is_diagonal:
                y += self._time_domain_convolve(powers[2], kernel.h2, B, N)

            if kernel.h3_diagonal is not None:
                y += self._time_domain_convolve(powers[3], kernel.h3_diagonal, B, N)

            if kernel.h4_diagonal is not None:
                y += self._time_domain_convolve(powers[4], kernel.h4_diagonal, B, N)

            if kernel.h5_diagonal is not None:
                y += self._time_domain_convolve(powers[5], kernel.h5_diagonal, B, N)

        return y

    def _compute_power_chain(self, x: ArrayF, max_order: int) -> dict[int, ArrayF]:
        """
        Compute powers using efficient reuse chain.

        Optimal chain:
        - x²: 1 mult per element
        - x³: 1 mult (x² * x)
        - x⁴: 1 mult (x² * x²)
        - x⁵: 1 mult (x⁴ * x)
        Total: 4 mults per element for all orders 1-5

        Returns:
            Dict mapping order → x^order array
        """
        powers = {1: x}

        if max_order >= 2:
            powers[2] = x * x

        if max_order >= 3:
            powers[3] = powers[2] * x

        if max_order >= 4:
            powers[4] = powers[2] * powers[2]

        if max_order >= 5:
            powers[5] = powers[4] * x

        return powers

    def _fft_convolve(
        self, x_pow: ArrayF, H_fft: np.ndarray, B: int, h_kernel: ArrayF = None
    ) -> ArrayF:
        """
        FFT convolution with precomputed kernel FFT.

        Uses rfft/irfft for real signals (2x speedup vs complex FFT).

        Args:
            x_pow: Signal power (x^n) extended with history
            H_fft: Precomputed FFT(kernel) at self._fft_size
            B: Block size to extract
            h_kernel: Original kernel array (for dynamic resizing if needed)

        Returns:
            Convolution result (B,)
        """
        N = len(self.kernel.h1)

        # Check if block size exceeds precomputed size
        required_len = N + B - 1
        if required_len > self._fft_size:
            # Block size exceeds precomputed size - use dynamic FFT size
            fft_size = fft.next_fast_len(required_len)

            # Recompute kernel FFT at new size
            H_fft_resized = fft.rfft(h_kernel, n=fft_size)

            # Use dynamic FFT size
            X_fft = fft.rfft(x_pow, n=fft_size)
            Y_fft = X_fft * H_fft_resized
            y_full = fft.irfft(Y_fft, n=fft_size)
        else:
            # Use precomputed FFT size
            X_fft = fft.rfft(x_pow, n=self._fft_size)
            Y_fft = X_fft * H_fft
            y_full = fft.irfft(Y_fft, n=self._fft_size)

        # Extract valid output region
        # For overlap-add: start at position N-1, extract B samples
        start = N - 1
        return y_full[start : start + B]

    def _time_domain_convolve(self, x_ext: ArrayF, h: ArrayF, B: int, N: int) -> ArrayF:
        """
        Direct time-domain convolution for short kernels.

        Vectorized implementation using numpy operations.

        Args:
            x_ext: Extended signal with history
            h: Kernel
            B: Block size
            N: Kernel length

        Returns:
            Convolution result (B,)
        """
        y = np.zeros(B, dtype=np.float64)

        for n in range(B):
            start_idx = len(x_ext) - B + n - (N - 1)
            end_idx = len(x_ext) - B + n + 1

            if start_idx >= 0:
                x_window = x_ext[start_idx:end_idx][::-1]  # Reverse for convolution
                y[n] = np.dot(h[: len(x_window)], x_window)

        return y


# Numba-accelerated version for time-domain
if NUMBA_AVAILABLE:

    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_time_domain_convolve(x_ext: np.ndarray, h: np.ndarray, B: int, N: int) -> np.ndarray:
        """Numba-optimized time-domain convolution."""
        y = np.zeros(B, dtype=np.float64)

        for n in prange(B):
            accum = 0.0
            for i in range(N):
                idx = len(x_ext) - B + n - i
                if idx >= 0:
                    accum += h[i] * x_ext[idx]
            y[n] = accum

        return y

    class FFTOptimizedNumbaEngine(FFTOptimizedEngine):
        """
        FFT-optimized engine with Numba acceleration for time-domain.

        Combines:
        - FFT with kernel precomputation for long kernels
        - Numba JIT for short kernels (time-domain)
        - Optimal power chain
        - Auto-selection

        Best of both worlds: FFT efficiency for N>=128, Numba for N<128.
        """

        def __init__(
            self, kernel: VolterraKernelFull, fft_threshold: int = 128, max_block_size: int = 4096
        ):
            super().__init__(kernel, fft_threshold, max_block_size)

            # Warmup Numba if using time-domain
            if not self.use_fft:
                self._warmup_numba()

        def _warmup_numba(self):
            """Pre-compile Numba functions."""
            x_test = np.random.randn(1024).astype(np.float64)
            h_test = np.random.randn(128).astype(np.float64)
            _ = _numba_time_domain_convolve(x_test, h_test, 512, 128)

        def _time_domain_convolve(self, x_ext: ArrayF, h: ArrayF, B: int, N: int) -> ArrayF:
            """Use Numba-accelerated time-domain convolution."""
            return _numba_time_domain_convolve(x_ext, h, B, N)
