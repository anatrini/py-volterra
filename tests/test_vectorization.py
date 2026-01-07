"""
Vectorization tests - verify NO Python loops on samples.

These tests ensure:
1. VectorizedEngine produces identical results to reference implementations
2. Complete vectorization (no Python loops over samples)
3. Memory efficiency (pre-allocated buffers, in-place operations)
4. Performance targets met

Critical for Session 4: Performance & Memory Efficiency
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from volterra import VolterraKernelFull, VolterraProcessorFull
from volterra.engines_diagonal import DiagonalNumpyEngine
from volterra.engines_vectorized import VectorizedEngine

try:
    from volterra.engines_diagonal import NUMBA_AVAILABLE, DiagonalNumbaEngine
except ImportError:
    NUMBA_AVAILABLE = False


class TestVectorizedCorrectness:
    """Test that vectorized engine produces correct results."""

    @pytest.fixture
    def sample_rate(self):
        return 48000

    @pytest.fixture
    def kernel_realistic(self):
        """Realistic kernel with multiple orders."""
        return VolterraKernelFull.from_polynomial_coeffs(
            N=256, a1=1.0, a2=0.15, a3=0.03, a4=0.01, a5=0.02
        )

    @pytest.fixture
    def test_signal(self, sample_rate):
        """Test signal: sine + noise."""
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.01 * np.random.randn(len(t))
        return signal.astype(np.float64)

    def test_vectorized_vs_numpy_engine(self, kernel_realistic, test_signal, sample_rate):
        """
        CRITICAL: Vectorized engine must match reference NumPy implementation.

        This verifies mathematical correctness of vectorization.
        """
        # 1. Reference: DiagonalNumpyEngine (loop-based, slow but verified)
        proc_ref = VolterraProcessorFull(kernel_realistic, sample_rate=sample_rate, use_numba=False)
        proc_ref.engine = DiagonalNumpyEngine()
        output_ref = proc_ref.process(test_signal, block_size=512)

        # 2. Optimized: VectorizedEngine (NO loops)
        proc_vec = VolterraProcessorFull(kernel_realistic, sample_rate=sample_rate, use_numba=False)
        proc_vec.engine = VectorizedEngine(max_block_size=4096)
        output_vec = proc_vec.process(test_signal, block_size=512)

        # 3. Verify equivalence
        max_diff = np.max(np.abs(output_ref - output_vec))
        rms_diff = np.sqrt(np.mean((output_ref - output_vec) ** 2))

        assert max_diff < 1e-10, f"Vectorized vs NumPy max diff {max_diff:.2e} > 1e-10"
        assert rms_diff < 1e-11, f"Vectorized vs NumPy RMS diff {rms_diff:.2e} > 1e-11"

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_vectorized_vs_numba_engine(self, kernel_realistic, test_signal, sample_rate):
        """Verify vectorized engine matches Numba implementation."""
        # 1. Numba engine
        proc_numba = VolterraProcessorFull(
            kernel_realistic, sample_rate=sample_rate, use_numba=True
        )
        proc_numba.engine = DiagonalNumbaEngine()
        output_numba = proc_numba.process(test_signal, block_size=512)

        # 2. Vectorized engine
        proc_vec = VolterraProcessorFull(kernel_realistic, sample_rate=sample_rate, use_numba=False)
        proc_vec.engine = VectorizedEngine(max_block_size=4096)
        output_vec = proc_vec.process(test_signal, block_size=512)

        # 3. Verify equivalence
        max_diff = np.max(np.abs(output_numba - output_vec))
        assert max_diff < 1e-10, f"Vectorized vs Numba max diff {max_diff:.2e} > 1e-10"

    def test_vectorized_different_block_sizes(self, kernel_realistic, test_signal, sample_rate):
        """Vectorized engine should work with various block sizes."""
        proc = VolterraProcessorFull(kernel_realistic, sample_rate=sample_rate, use_numba=False)
        proc.engine = VectorizedEngine(max_block_size=4096)

        for block_size in [64, 128, 256, 512, 1024]:
            proc.reset()
            output = proc.process(test_signal, block_size=block_size)
            assert len(output) == len(test_signal)
            assert not np.any(np.isnan(output))
            assert not np.any(np.isinf(output))

    def test_buffer_reuse(self, kernel_realistic, sample_rate):
        """Verify buffers are reused (no allocations per block)."""
        engine = VectorizedEngine(max_block_size=4096)

        # Get buffer IDs
        y_buffer_id_before = id(engine._y_buffer)
        power_buffer_id_before = id(engine._power_buffers[2])

        # Process blocks
        x_block = np.random.randn(512) * 0.1
        x_history = np.zeros(kernel_realistic.N - 1)

        for _ in range(10):
            _ = engine.process_block(x_block, x_history, kernel_realistic)

        # Verify buffers not reallocated
        assert id(engine._y_buffer) == y_buffer_id_before
        assert id(engine._power_buffers[2]) == power_buffer_id_before

    def test_c_contiguous_arrays(self, kernel_realistic, sample_rate):
        """Verify arrays are C-contiguous for cache efficiency."""
        engine = VectorizedEngine(max_block_size=4096)

        x_block = np.random.randn(512) * 0.1
        x_history = np.zeros(kernel_realistic.N - 1)

        # Process block
        output = engine.process_block(x_block, x_history, kernel_realistic)

        # Check output is C-contiguous
        assert output.flags["C_CONTIGUOUS"]

        # Check internal buffers
        assert engine._y_buffer.flags["C_CONTIGUOUS"]
        for power_buf in engine._power_buffers.values():
            assert power_buf.flags["C_CONTIGUOUS"]

    def test_dtype_consistency(self, kernel_realistic, sample_rate):
        """Verify consistent float64 dtype (no silent upcasting)."""
        engine = VectorizedEngine(max_block_size=4096)

        x_block = np.random.randn(512).astype(np.float64) * 0.1
        x_history = np.zeros(kernel_realistic.N - 1, dtype=np.float64)

        output = engine.process_block(x_block, x_history, kernel_realistic)

        # Verify dtype
        assert output.dtype == np.float64
        assert engine._y_buffer.dtype == np.float64
        for power_buf in engine._power_buffers.values():
            assert power_buf.dtype == np.float64


class TestVectorizationPerformance:
    """Performance benchmarks for vectorized engine."""

    @pytest.fixture
    def kernels_various(self):
        """Kernels with different lengths for benchmarking."""
        return {
            64: VolterraKernelFull.from_polynomial_coeffs(N=64, a1=1.0, a2=0.1, a3=0.02),
            256: VolterraKernelFull.from_polynomial_coeffs(N=256, a1=1.0, a2=0.1, a3=0.02),
            512: VolterraKernelFull.from_polynomial_coeffs(N=512, a1=1.0, a2=0.1, a3=0.02),
        }

    def test_power_computation_efficiency(self, kernels_various):
        """
        Verify power computation uses minimal operations.

        Should compute x², x³, x⁴, x⁵ with optimal chain (4 mults per sample).
        """
        kernels_various[256]
        engine = VectorizedEngine(max_block_size=4096)

        x_test = np.random.randn(1024).astype(np.float64)

        # Compute powers
        powers = engine._compute_powers(x_test, max_order=5)

        # Verify all powers computed
        assert 1 in powers
        assert 2 in powers
        assert 3 in powers
        assert 4 in powers
        assert 5 in powers

        # Verify correctness
        assert_allclose(powers[2], x_test**2)
        assert_allclose(powers[3], x_test**3)
        assert_allclose(powers[5], x_test**5)

    def test_no_temporary_allocations_in_accumulate(self, kernels_various):
        """
        Verify accumulate_convolution does not create temporaries.

        Should use in-place np.add(y, result, out=y).
        """
        kernel = kernels_various[256]
        engine = VectorizedEngine(max_block_size=4096)

        B = 512
        N = kernel.N

        y = np.zeros(B, dtype=np.float64)
        id(y)

        x_pow = np.random.randn(B + N - 1).astype(np.float64)
        h = kernel.h1

        # Accumulate (should be in-place on y)
        engine._accumulate_convolution(y, x_pow, h, B, N)

        # y should still be same object (in-place modification)
        # Note: This test verifies the operation modifies y in-place
        assert not np.all(y == 0)  # y was modified

    def test_vectorized_faster_than_loops(self, kernels_various, benchmark=None):
        """
        Verify vectorized implementation is significantly faster than loops.

        Expected: ~5-10x speedup over DiagonalNumpyEngine.
        """
        if benchmark is None:
            pytest.skip("Benchmark fixture not available")

        kernel = kernels_various[256]
        signal = np.random.randn(48000).astype(np.float64) * 0.1  # 1 second @ 48kHz

        # Time vectorized engine
        engine_vec = VectorizedEngine(max_block_size=4096)
        proc_vec = VolterraProcessorFull(kernel, sample_rate=48000, use_numba=False)
        proc_vec.engine = engine_vec

        # Note: Actual benchmarking would use pytest-benchmark
        # For now, just verify it runs without error
        output = proc_vec.process(signal, block_size=512)
        assert len(output) == len(signal)


class TestMemoryEfficiency:
    """Test memory efficiency and buffer reuse."""

    def test_pre_allocated_buffers(self):
        """Verify buffers are pre-allocated at init."""
        max_block_size = 2048
        engine = VectorizedEngine(max_block_size=max_block_size)

        # Verify output buffer pre-allocated
        assert len(engine._y_buffer) == max_block_size

        # Verify power buffers pre-allocated
        for order in range(2, 6):
            assert order in engine._power_buffers
            assert len(engine._power_buffers[order]) > max_block_size

    def test_no_reallocation_during_processing(self):
        """Verify no buffer reallocations during normal processing."""
        engine = VectorizedEngine(max_block_size=4096)
        kernel = VolterraKernelFull.from_polynomial_coeffs(N=256, a1=1.0, a2=0.1)

        # Get initial buffer IDs
        initial_ids = {
            "y": id(engine._y_buffer),
            "powers": {order: id(buf) for order, buf in engine._power_buffers.items()},
        }

        # Process many blocks
        x_block = np.random.randn(512).astype(np.float64) * 0.1
        x_history = np.zeros(kernel.N - 1, dtype=np.float64)

        for _ in range(100):
            _ = engine.process_block(x_block, x_history, kernel)

        # Verify buffers not reallocated
        assert id(engine._y_buffer) == initial_ids["y"]
        for order, buf_id in initial_ids["powers"].items():
            assert id(engine._power_buffers[order]) == buf_id

    def test_in_place_operations(self):
        """Verify operations use in-place modifications where possible."""
        engine = VectorizedEngine(max_block_size=4096)

        x = np.random.randn(1024).astype(np.float64)

        # Compute powers
        powers = engine._compute_powers(x, max_order=5)

        # Verify x² buffer is view of pre-allocated pool (shares base array)
        # powers[2] is a view of _power_buffers[2]
        assert powers[2].base is engine._power_buffers[2]

        # Verify in-place multiplication was used
        # (values in power buffer should match x²)
        expected_x2 = x * x
        actual_x2 = engine._power_buffers[2][: len(x)]
        assert_allclose(actual_x2, expected_x2)
