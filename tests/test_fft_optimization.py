"""
FFT optimization tests.

These tests verify:
1. FFT engines work correctly
2. Auto-selection between time-domain and FFT
3. FFT kernel precomputation
4. Mathematical equivalence with time-domain
5. Performance characteristics (FFT vs time-domain crossover)

Critical for Session 3: Algorithm Selection & FFT Optimization
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from volterra import VolterraKernelFull, VolterraProcessorFull

try:
    from volterra.engines_fft import FFTOptimizedEngine, FFTOptimizedNumbaEngine, NUMBA_AVAILABLE
    FFT_ENGINES_AVAILABLE = True
except ImportError:
    FFT_ENGINES_AVAILABLE = False
    pytest.skip("FFT engines not available", allow_module_level=True)

from volterra.engines_diagonal import DiagonalNumpyEngine


class TestFFTEngineCorrectness:
    """Test that FFT engines produce mathematically correct results."""

    @pytest.fixture
    def sample_rate(self):
        return 48000

    @pytest.fixture
    def kernel_long(self):
        """Long kernel (N=512) where FFT is beneficial."""
        return VolterraKernelFull.from_polynomial_coeffs(
            N=512,
            a1=1.0,
            a2=0.15,
            a3=0.03,
            a4=0.01,
            a5=0.02
        )

    @pytest.fixture
    def kernel_short(self):
        """Short kernel (N=64) where time-domain is faster."""
        return VolterraKernelFull.from_polynomial_coeffs(
            N=64,
            a1=1.0,
            a2=0.15,
            a3=0.03
        )

    @pytest.fixture
    def test_signal(self, sample_rate):
        """Test signal: swept sine + noise."""
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Chirp from 100 Hz to 5 kHz
        f0, f1 = 100, 5000
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
        signal = 0.2 * np.sin(phase) + 0.01 * np.random.randn(len(t))
        return signal.astype(np.float64)

    def test_fft_engine_vs_time_domain(self, kernel_long, test_signal, sample_rate):
        """
        CRITICAL: FFT engine must produce identical results to time-domain.

        This verifies mathematical correctness of:
        - FFT kernel precomputation
        - rfft/irfft usage
        - Overlap-add extraction
        """
        # 1. Process with time-domain engine
        proc_time = VolterraProcessorFull(kernel_long, sample_rate=sample_rate, use_numba=False)
        proc_time.engine = DiagonalNumpyEngine()  # Force time-domain
        output_time = proc_time.process(test_signal, block_size=512)

        # 2. Process with FFT engine
        proc_fft = VolterraProcessorFull(kernel_long, sample_rate=sample_rate, use_numba=False)
        proc_fft.engine = FFTOptimizedEngine(kernel_long, fft_threshold=128)
        output_fft = proc_fft.process(test_signal, block_size=512)

        # 3. Verify equivalence
        max_diff = np.max(np.abs(output_time - output_fft))
        rms_diff = np.sqrt(np.mean((output_time - output_fft)**2))

        assert max_diff < 1e-10, f"FFT vs time-domain max diff {max_diff:.2e} > 1e-10"
        assert rms_diff < 1e-11, f"FFT vs time-domain RMS diff {rms_diff:.2e} > 1e-11"

    def test_fft_kernel_precomputation(self, kernel_long):
        """Verify that FFT kernels are precomputed correctly."""
        engine = FFTOptimizedEngine(kernel_long, fft_threshold=128)

        # Should have precomputed FFT kernels
        assert engine.use_fft is True
        assert engine._fft_kernels is not None
        assert 'h1' in engine._fft_kernels
        assert 'h2' in engine._fft_kernels
        assert 'h3' in engine._fft_kernels
        assert 'h5' in engine._fft_kernels

        # FFT size should be >= N + B_max - 1
        N = kernel_long.N
        B_max = engine.max_block_size
        assert engine._fft_size >= N + B_max - 1

        # Check FFT kernels have correct shape (rfft output)
        expected_fft_len = engine._fft_size // 2 + 1
        assert len(engine._fft_kernels['h1']) == expected_fft_len

    def test_auto_selection_long_kernel(self, kernel_long, sample_rate):
        """Processor should auto-select FFT for long kernels (N>=128)."""
        proc = VolterraProcessorFull(kernel_long, sample_rate=sample_rate, use_numba=False)

        # Should have selected FFT engine
        assert isinstance(proc.engine, FFTOptimizedEngine)
        assert proc.engine.use_fft is True

    def test_auto_selection_short_kernel(self, kernel_short, sample_rate):
        """Processor should use time-domain for short kernels (N<128)."""
        proc = VolterraProcessorFull(kernel_short, sample_rate=sample_rate, use_numba=False)

        # Should have selected time-domain engine
        assert isinstance(proc.engine, DiagonalNumpyEngine)

    def test_fft_engine_different_block_sizes(self, kernel_long, test_signal, sample_rate):
        """FFT engine should work correctly with different block sizes."""
        proc = VolterraProcessorFull(kernel_long, sample_rate=sample_rate, use_numba=False)
        proc.engine = FFTOptimizedEngine(kernel_long, fft_threshold=128, max_block_size=4096)

        # Test different block sizes
        for block_size in [128, 256, 512, 1024]:
            proc.reset()
            output = proc.process(test_signal, block_size=block_size)
            assert len(output) == len(test_signal)
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))

    def test_fft_size_optimization(self, kernel_long):
        """Verify next_fast_len is used for optimal FFT size."""
        from scipy.fft import next_fast_len

        engine = FFTOptimizedEngine(kernel_long, fft_threshold=128, max_block_size=2048)

        N = kernel_long.N
        B_max = 2048
        required_len = N + B_max - 1

        # FFT size should be next_fast_len of required
        expected_size = next_fast_len(required_len)
        assert engine._fft_size == expected_size

    def test_rfft_usage(self, kernel_long):
        """Verify that rfft is used (not complex FFT)."""
        engine = FFTOptimizedEngine(kernel_long, fft_threshold=128)

        # Precomputed FFT should be complex with length N//2+1 (rfft output)
        fft_len = engine._fft_size // 2 + 1
        h1_fft = engine._fft_kernels['h1']

        assert len(h1_fft) == fft_len
        assert h1_fft.dtype == np.complex128  # rfft output is complex

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_fft_numba_engine(self, kernel_long, test_signal, sample_rate):
        """Test FFT+Numba hybrid engine."""
        proc = VolterraProcessorFull(kernel_long, sample_rate=sample_rate, use_numba=True)

        # Should auto-select FFT+Numba hybrid
        assert isinstance(proc.engine, FFTOptimizedNumbaEngine)

        # Process signal
        output = proc.process(test_signal, block_size=512)
        assert len(output) == len(test_signal)
        assert not np.any(np.isnan(output))

    def test_power_chain_optimization(self, kernel_long, sample_rate):
        """Verify power chain is still used with FFT engine."""
        proc = VolterraProcessorFull(kernel_long, sample_rate=sample_rate, use_numba=False)
        proc.engine = FFTOptimizedEngine(kernel_long, fft_threshold=128)

        x_test = np.random.randn(1024) * 0.1
        x_ext = np.concatenate([np.zeros(kernel_long.N - 1), x_test])

        # Compute power chain
        powers = proc.engine._compute_power_chain(x_ext, kernel_long.max_order)

        # Verify powers are computed correctly
        assert 1 in powers
        assert 2 in powers
        assert 3 in powers
        assert 5 in powers

        # Verify power correctness
        assert_allclose(powers[2], x_ext ** 2)
        assert_allclose(powers[3], x_ext ** 3)
        assert_allclose(powers[5], x_ext ** 5)


class TestFFTPerformance:
    """Test FFT vs time-domain performance characteristics."""

    @pytest.fixture
    def kernels_various_lengths(self):
        """Kernels with different lengths for crossover testing."""
        lengths = [32, 64, 128, 256, 512]
        return {
            N: VolterraKernelFull.from_polynomial_coeffs(N=N, a1=1.0, a2=0.1, a3=0.02)
            for N in lengths
        }

    def test_fft_threshold_behavior(self, kernels_various_lengths):
        """
        Verify that auto-selection threshold works correctly.

        Threshold is set at N=128:
        - N < 128: use time-domain
        - N >= 128: use FFT
        """
        for N, kernel in kernels_various_lengths.items():
            engine = FFTOptimizedEngine(kernel, fft_threshold=128)

            if N < 128:
                assert engine.use_fft is False, f"N={N} should use time-domain"
            else:
                assert engine.use_fft is True, f"N={N} should use FFT"

    def test_no_fft_recomputation(self, kernels_various_lengths):
        """
        CRITICAL: FFT(kernel) must be precomputed, not recomputed per block.

        This is a key optimization from Session 3.
        """
        kernel = kernels_various_lengths[512]
        engine = FFTOptimizedEngine(kernel, fft_threshold=128)

        # Get precomputed FFT
        h1_fft_before = engine._fft_kernels['h1'].copy()

        # Process a block
        x_block = np.random.randn(512) * 0.1
        x_history = np.zeros(kernel.N - 1)
        _ = engine.process_block(x_block, x_history, kernel)

        # FFT kernel should be unchanged (precomputed)
        h1_fft_after = engine._fft_kernels['h1']
        assert_allclose(h1_fft_before, h1_fft_after)


class TestFFTEdgeCases:
    """Test edge cases and error handling."""

    def test_very_short_kernel(self):
        """FFT engine should handle very short kernels (N=16)."""
        kernel = VolterraKernelFull.from_polynomial_coeffs(N=16, a1=1.0, a2=0.1)
        engine = FFTOptimizedEngine(kernel, fft_threshold=128)

        # Should use time-domain
        assert engine.use_fft is False

        # Should still work
        x_block = np.random.randn(256) * 0.1
        x_history = np.zeros(kernel.N - 1)
        y = engine.process_block(x_block, x_history, kernel)
        assert len(y) == len(x_block)

    def test_large_block_size(self):
        """FFT engine should handle large block sizes."""
        kernel = VolterraKernelFull.from_polynomial_coeffs(N=256, a1=1.0, a2=0.1)
        engine = FFTOptimizedEngine(kernel, fft_threshold=128, max_block_size=8192)

        # Process large block
        x_block = np.random.randn(8192) * 0.1
        x_history = np.zeros(kernel.N - 1)
        y = engine.process_block(x_block, x_history, kernel)
        assert len(y) == len(x_block)
        assert not np.any(np.isnan(y))
