"""
Comprehensive tests for engines_optimized module.

Tests both OptimizedDiagonalEngine and OptimizedNumbaEngine (if available).
"""

import numpy as np
import pytest

from volterra.engines_optimized import NUMBA_AVAILABLE, OptimizedDiagonalEngine
from volterra.kernels_full import VolterraKernelFull

if NUMBA_AVAILABLE:
    from volterra.engines_optimized import OptimizedNumbaEngine


class TestOptimizedDiagonalEngine:
    """Test OptimizedDiagonalEngine."""

    def test_init_default_threshold(self):
        """Test initialization with default FFT threshold."""
        engine = OptimizedDiagonalEngine()
        assert engine.use_fft_threshold == 256

    def test_init_custom_threshold(self):
        """Test initialization with custom FFT threshold."""
        engine = OptimizedDiagonalEngine(use_fft_threshold=512)
        assert engine.use_fft_threshold == 512

    def test_linear_only_kernel(self):
        """Test processing with linear kernel only."""
        np.random.seed(42)
        engine = OptimizedDiagonalEngine()

        # Linear kernel
        h1 = np.array([0.5, 0.3, 0.2])
        kernel = VolterraKernelFull(h1=h1)

        # Input block and history
        x_block = np.random.randn(10) * 0.1
        x_history = np.random.randn(2) * 0.1

        y = engine.process_block(x_block, x_history, kernel)

        assert y.shape == (10,)
        assert np.all(np.isfinite(y))

    def test_linear_and_quadratic_diagonal(self):
        """Test processing with linear and diagonal quadratic kernels."""
        np.random.seed(42)
        engine = OptimizedDiagonalEngine()

        h1 = np.array([0.5, 0.3, 0.2])
        h2_diag = np.array([0.1, 0.05, 0.02])
        kernel = VolterraKernelFull(h1=h1, h2=h2_diag, h2_is_diagonal=True)

        x_block = np.random.randn(10) * 0.1
        x_history = np.random.randn(2) * 0.1

        y = engine.process_block(x_block, x_history, kernel)

        assert y.shape == (10,)
        assert np.all(np.isfinite(y))

    def test_full_h2_matrix(self):
        """Test processing with full h2 matrix (backward compatibility)."""
        np.random.seed(42)
        engine = OptimizedDiagonalEngine()

        N = 3
        h1 = np.array([0.5, 0.3, 0.2])
        h2_full = np.random.randn(N, N) * 0.1
        h2_full = 0.5 * (h2_full + h2_full.T)  # Make symmetric
        kernel = VolterraKernelFull(h1=h1, h2=h2_full, h2_is_diagonal=False)

        x_block = np.random.randn(10) * 0.1
        x_history = np.random.randn(N - 1) * 0.1

        y = engine.process_block(x_block, x_history, kernel)

        assert y.shape == (10,)
        assert np.all(np.isfinite(y))

    def test_third_order_diagonal(self):
        """Test with 3rd order diagonal kernel."""
        np.random.seed(42)
        engine = OptimizedDiagonalEngine()

        h1 = np.array([0.5, 0.3, 0.2])
        h2_diag = np.array([0.1, 0.05, 0.02])
        h3_diag = np.array([0.05, 0.02, 0.01])

        kernel = VolterraKernelFull(
            h1=h1, h2=h2_diag, h2_is_diagonal=True, h3_diagonal=h3_diag
        )

        x_block = np.random.randn(10) * 0.1
        x_history = np.random.randn(2) * 0.1

        y = engine.process_block(x_block, x_history, kernel)

        assert y.shape == (10,)
        assert np.all(np.isfinite(y))

    def test_all_orders_up_to_5(self):
        """Test with all orders up to 5th."""
        np.random.seed(42)
        engine = OptimizedDiagonalEngine()

        h1 = np.array([0.5, 0.3, 0.2])
        h2_diag = np.array([0.1, 0.05, 0.02])
        h3_diag = np.array([0.05, 0.02, 0.01])
        h4_diag = np.array([0.02, 0.01, 0.005])
        h5_diag = np.array([0.01, 0.005, 0.002])

        kernel = VolterraKernelFull(
            h1=h1,
            h2=h2_diag,
            h2_is_diagonal=True,
            h3_diagonal=h3_diag,
            h4_diagonal=h4_diag,
            h5_diagonal=h5_diag,
        )

        x_block = np.random.randn(10) * 0.1
        x_history = np.random.randn(2) * 0.1

        y = engine.process_block(x_block, x_history, kernel)

        assert y.shape == (10,)
        assert np.all(np.isfinite(y))

    def test_fft_convolution_long_kernel(self):
        """Test FFT convolution path with long kernel."""
        np.random.seed(42)
        # Use low threshold to force FFT
        engine = OptimizedDiagonalEngine(use_fft_threshold=5)

        # Long kernel (N > threshold)
        N = 10
        h1 = np.random.randn(N) * 0.1

        kernel = VolterraKernelFull(h1=h1)

        x_block = np.random.randn(20) * 0.1
        x_history = np.random.randn(N - 1) * 0.1

        y = engine.process_block(x_block, x_history, kernel)

        assert y.shape == (20,)
        assert np.all(np.isfinite(y))

    def test_direct_convolution_short_kernel(self):
        """Test direct time-domain convolution with short kernel."""
        np.random.seed(42)
        # Use high threshold to force direct convolution
        engine = OptimizedDiagonalEngine(use_fft_threshold=1000)

        h1 = np.array([0.5, 0.3, 0.2])
        kernel = VolterraKernelFull(h1=h1)

        x_block = np.random.randn(10) * 0.1
        x_history = np.random.randn(2) * 0.1

        y = engine.process_block(x_block, x_history, kernel)

        assert y.shape == (10,)
        assert np.all(np.isfinite(y))

    def test_power_chain_computation(self):
        """Test _compute_power_chain method."""
        engine = OptimizedDiagonalEngine()

        x = np.array([2.0, 3.0, 1.0])

        # Test orders 1-5
        powers = engine._compute_power_chain(x, max_order=5)

        np.testing.assert_array_equal(powers[1], x)
        np.testing.assert_array_almost_equal(powers[2], x**2)
        np.testing.assert_array_almost_equal(powers[3], x**3)
        np.testing.assert_array_almost_equal(powers[4], x**4)
        np.testing.assert_array_almost_equal(powers[5], x**5)

    def test_power_chain_various_orders(self):
        """Test power chain for various max orders."""
        engine = OptimizedDiagonalEngine()
        x = np.array([2.0, 3.0])

        for max_order in [1, 2, 3, 4, 5]:
            powers = engine._compute_power_chain(x, max_order)
            assert len(powers) == max_order
            assert 1 in powers

    def test_different_block_sizes(self):
        """Test with different block sizes."""
        np.random.seed(42)
        engine = OptimizedDiagonalEngine()

        h1 = np.array([0.5, 0.3, 0.2])
        kernel = VolterraKernelFull(h1=h1)

        for B in [5, 10, 50, 100]:
            x_block = np.random.randn(B) * 0.1
            x_history = np.random.randn(2) * 0.1

            y = engine.process_block(x_block, x_history, kernel)
            assert y.shape == (B,)

    def test_consistency_fft_vs_direct(self):
        """Test that FFT and direct convolution produce same results."""
        np.random.seed(42)

        # Same kernel, different thresholds
        h1 = np.random.randn(10) * 0.1
        kernel = VolterraKernelFull(h1=h1)

        x_block = np.random.randn(20) * 0.1
        x_history = np.random.randn(9) * 0.1

        # Force FFT
        engine_fft = OptimizedDiagonalEngine(use_fft_threshold=5)
        y_fft = engine_fft.process_block(x_block, x_history, kernel)

        # Force direct
        engine_direct = OptimizedDiagonalEngine(use_fft_threshold=1000)
        y_direct = engine_direct.process_block(x_block, x_history, kernel)

        # Should be very close (minor numerical differences acceptable)
        np.testing.assert_allclose(y_fft, y_direct, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
class TestOptimizedNumbaEngine:
    """Test OptimizedNumbaEngine (if Numba is available)."""

    def test_init_warmup(self):
        """Test that initialization performs warmup."""
        # Should not raise
        engine = OptimizedNumbaEngine()
        assert engine is not None

    def test_linear_kernel(self):
        """Test with linear kernel."""
        np.random.seed(42)
        engine = OptimizedNumbaEngine()

        h1 = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        kernel = VolterraKernelFull(h1=h1)

        x_block = np.random.randn(10).astype(np.float64) * 0.1
        x_history = np.random.randn(2).astype(np.float64) * 0.1

        y = engine.process_block(x_block, x_history, kernel)

        assert y.shape == (10,)
        assert y.dtype == np.float64
        assert np.all(np.isfinite(y))

    def test_all_orders(self):
        """Test with all diagonal orders."""
        np.random.seed(42)
        engine = OptimizedNumbaEngine()

        h1 = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        h2_diag = np.array([0.1, 0.05, 0.02], dtype=np.float64)
        h3_diag = np.array([0.05, 0.02, 0.01], dtype=np.float64)
        h4_diag = np.array([0.02, 0.01, 0.005], dtype=np.float64)
        h5_diag = np.array([0.01, 0.005, 0.002], dtype=np.float64)

        kernel = VolterraKernelFull(
            h1=h1,
            h2=h2_diag,
            h2_is_diagonal=True,
            h3_diagonal=h3_diag,
            h4_diagonal=h4_diag,
            h5_diagonal=h5_diag,
        )

        x_block = np.random.randn(10).astype(np.float64) * 0.1
        x_history = np.random.randn(2).astype(np.float64) * 0.1

        y = engine.process_block(x_block, x_history, kernel)

        assert y.shape == (10,)
        assert np.all(np.isfinite(y))

    def test_consistency_with_optimized_diagonal(self):
        """Test that Numba engine produces same results as optimized diagonal."""
        np.random.seed(42)

        h1 = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        h2_diag = np.array([0.1, 0.05, 0.02], dtype=np.float64)
        kernel = VolterraKernelFull(h1=h1, h2=h2_diag, h2_is_diagonal=True)

        x_block = np.random.randn(10).astype(np.float64) * 0.1
        x_history = np.random.randn(2).astype(np.float64) * 0.1

        # Numba engine
        engine_numba = OptimizedNumbaEngine()
        y_numba = engine_numba.process_block(x_block, x_history, kernel)

        # Optimized diagonal (force direct convolution for consistency)
        engine_opt = OptimizedDiagonalEngine(use_fft_threshold=1000)
        y_opt = engine_opt.process_block(x_block, x_history, kernel)

        # Should be very close
        np.testing.assert_allclose(y_numba, y_opt, rtol=1e-10, atol=1e-12)

    def test_various_block_sizes(self):
        """Test Numba engine with various block sizes."""
        np.random.seed(42)
        engine = OptimizedNumbaEngine()

        h1 = np.array([0.5, 0.3], dtype=np.float64)
        kernel = VolterraKernelFull(h1=h1)

        for B in [5, 10, 50]:
            x_block = np.random.randn(B).astype(np.float64) * 0.1
            x_history = np.random.randn(1).astype(np.float64) * 0.1

            y = engine.process_block(x_block, x_history, kernel)
            assert y.shape == (B,)
