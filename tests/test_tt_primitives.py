"""
Tests for Tensor-Train (TT) primitives.

These tests verify:
1. TT core validation and representation
2. TT-matrix-vector multiplication
3. TT-to-full tensor materialization
4. Basic TT-ALS/MALS solver functionality

Critical for STEP 3: TT Primitives Implementation
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from volterra.tt import (
    TTTensor,
    validate_tt_cores,
    tt_matvec,
    tt_to_full,
    tt_als,
    tt_mals,
    TTALSConfig,
    TTMALSConfig,
)


class TestTTTensorValidation:
    """Test TT tensor representation and validation."""

    def test_valid_tt_tensor_creation(self):
        """Valid TT cores should create TTTensor."""
        cores = [
            np.random.randn(1, 4, 2),
            np.random.randn(2, 4, 3),
            np.random.randn(3, 4, 1),
        ]
        tt = TTTensor(cores)

        assert tt.ndim == 3
        assert tt.shape == (4, 4, 4)
        assert tt.ranks == (1, 2, 3, 1)

    def test_tt_tensor_single_core(self):
        """Single core (vector) should work."""
        cores = [np.random.randn(1, 10, 1)]
        tt = TTTensor(cores)

        assert tt.ndim == 1
        assert tt.shape == (10,)
        assert tt.ranks == (1, 1)

    def test_tt_tensor_invalid_boundary_left(self):
        """First core must have r_0=1."""
        cores = [
            np.random.randn(2, 4, 2),  # r_0=2, invalid
            np.random.randn(2, 4, 1),
        ]

        with pytest.raises(ValueError, match="r_0=1"):
            TTTensor(cores)

    def test_tt_tensor_invalid_boundary_right(self):
        """Last core must have r_K=1."""
        cores = [
            np.random.randn(1, 4, 2),
            np.random.randn(2, 4, 3),  # r_K=3, invalid
        ]

        with pytest.raises(ValueError, match="r_K=1"):
            TTTensor(cores)

    def test_tt_tensor_rank_mismatch(self):
        """Adjacent cores must have compatible ranks."""
        cores = [
            np.random.randn(1, 4, 2),  # r_1=2
            np.random.randn(3, 4, 1),  # r_1=3, mismatch!
        ]

        with pytest.raises(ValueError, match="Rank mismatch"):
            TTTensor(cores)

    def test_tt_tensor_not_3d(self):
        """Cores must be 3D arrays."""
        cores = [
            np.random.randn(4, 2),  # 2D, invalid
        ]

        with pytest.raises(ValueError, match="must be 3D"):
            TTTensor(cores)

    def test_tt_tensor_empty_cores(self):
        """Empty cores list should raise error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            TTTensor([])

    def test_tt_tensor_repr(self):
        """String representation should show shape and ranks."""
        cores = [
            np.random.randn(1, 3, 2),
            np.random.randn(2, 3, 1),
        ]
        tt = TTTensor(cores)
        repr_str = repr(tt)

        assert "shape=(3, 3)" in repr_str
        assert "ranks=(1, 2, 1)" in repr_str


class TestTTMatVec:
    """Test TT-matrix-vector multiplication."""

    def test_tt_matvec_scalar_output(self):
        """TT-matvec should return scalar."""
        cores = [
            np.random.randn(1, 5, 2),
            np.random.randn(2, 5, 1),
        ]
        x = np.random.randn(5)

        y = tt_matvec(cores, x)

        assert isinstance(y, (float, np.floating))

    def test_tt_matvec_linear_case(self):
        """Single core (linear) should match dot product."""
        N = 10
        h1 = np.random.randn(N)
        cores = [h1.reshape(1, N, 1)]
        x = np.random.randn(N)

        y_tt = tt_matvec(cores, x)
        y_expected = np.dot(h1, x)

        assert_allclose(y_tt, y_expected, rtol=1e-10)

    def test_tt_matvec_dimension_mismatch(self):
        """Input vector size must match core dimensions."""
        cores = [
            np.random.randn(1, 5, 2),
            np.random.randn(2, 5, 1),
        ]
        x_wrong = np.random.randn(8)  # Wrong size

        with pytest.raises(ValueError, match="dimension"):
            tt_matvec(cores, x_wrong)

    def test_tt_matvec_rank_one(self):
        """Rank-1 TT should match separable product."""
        N = 6
        # Rank-1 TT represents: h(i,j) = a(i) * b(j)
        a = np.random.randn(N)
        b = np.random.randn(N)

        cores = [
            a.reshape(1, N, 1),
            b.reshape(1, N, 1),
        ]
        x = np.random.randn(N)

        y_tt = tt_matvec(cores, x)

        # Expected: sum_i sum_j a(i)*b(j)*x(i)*x(j) = (a'x) * (b'x)
        y_expected = np.dot(a, x) * np.dot(b, x)

        assert_allclose(y_tt, y_expected, rtol=1e-10)

    def test_tt_matvec_three_cores(self):
        """3-core TT should work correctly."""
        cores = [
            np.random.randn(1, 4, 2),
            np.random.randn(2, 4, 3),
            np.random.randn(3, 4, 1),
        ]
        x = np.random.randn(4)

        y = tt_matvec(cores, x)

        # Just check it runs and returns scalar
        assert isinstance(y, (float, np.floating))

    def test_tt_matvec_invalid_cores(self):
        """Invalid cores should raise error."""
        cores = [np.random.randn(1, 5, 2)]  # Missing boundary r_K=1

        # Modify to break boundary condition
        cores[0] = np.random.randn(1, 5, 2)  # r_1=2, but no next core

        with pytest.raises(ValueError):
            # Create full tensor to trigger validation
            TTTensor(cores)


class TestTTToFull:
    """Test TT-to-full tensor materialization."""

    def test_tt_to_full_2d(self):
        """2D tensor from TT cores."""
        N = 3
        cores = [
            np.ones((1, N, 2)),
            np.ones((2, N, 1)),
        ]

        A = tt_to_full(cores)

        assert A.shape == (N, N)
        # All-ones cores should give all-2s tensor (sum over rank=2)
        assert_allclose(A, 2.0 * np.ones((N, N)))

    def test_tt_to_full_3d(self):
        """3D tensor from TT cores."""
        cores = [
            np.random.randn(1, 2, 2),
            np.random.randn(2, 3, 2),
            np.random.randn(2, 4, 1),
        ]

        A = tt_to_full(cores)

        assert A.shape == (2, 3, 4)

    def test_tt_to_full_consistency_with_matvec(self):
        """Full tensor contraction should match TT-matvec."""
        N = 4
        cores = [
            np.random.randn(1, N, 2),
            np.random.randn(2, N, 1),
        ]
        x = np.random.randn(N)

        # TT-matvec
        y_tt = tt_matvec(cores, x)

        # Full tensor contraction
        A = tt_to_full(cores)  # (N, N)
        y_full = np.einsum('ij,i,j->', A, x, x)  # sum_ij A[i,j] * x[i] * x[j]

        assert_allclose(y_tt, y_full, rtol=1e-10)

    def test_tt_to_full_1d(self):
        """1D tensor (vector) from single core."""
        N = 10
        v = np.random.randn(N)
        cores = [v.reshape(1, N, 1)]

        A = tt_to_full(cores)

        assert A.shape == (N,)
        assert_allclose(A, v)


class TestTTALS:
    """Test TT-ALS solver (basic functionality)."""

    def test_tt_als_basic_call(self):
        """TT-ALS should run without errors."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        cores, info = tt_als(
            x, y,
            memory_length=5,
            order=2,
            ranks=[1, 2, 1]
        )

        assert len(cores) == 2  # order=2
        assert cores[0].shape[1] == 5  # memory_length=5
        assert 'loss_history' in info
        assert 'converged' in info

    def test_tt_als_with_config(self):
        """TT-ALS should accept config."""
        x = np.random.randn(100)
        y = np.random.randn(100)

        config = TTALSConfig(
            max_iter=10,
            tol=1e-4,
            verbose=False,
            init_method='zeros'
        )

        cores, info = tt_als(
            x, y,
            memory_length=5,
            order=2,
            ranks=[1, 2, 1],
            config=config
        )

        assert len(cores) == 2

    def test_tt_als_invalid_ranks(self):
        """Invalid ranks should raise error."""
        x = np.random.randn(100)
        y = np.random.randn(100)

        # Wrong number of ranks
        with pytest.raises(ValueError):
            tt_als(x, y, memory_length=5, order=2, ranks=[1, 2])

        # Invalid boundary ranks
        with pytest.raises(ValueError):
            tt_als(x, y, memory_length=5, order=2, ranks=[2, 2, 1])

    def test_tt_als_invalid_input_shape(self):
        """3D input should raise error."""
        x = np.random.randn(100, 2, 3)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="must be 1D or 2D"):
            tt_als(x, y, memory_length=5, order=2, ranks=[1, 2, 1])

    def test_tt_als_invalid_output_shape(self):
        """2D output should raise error."""
        x = np.random.randn(100)
        y = np.random.randn(100, 2)

        with pytest.raises(ValueError, match="must be 1D"):
            tt_als(x, y, memory_length=5, order=2, ranks=[1, 2, 1])


class TestTTMALS:
    """Test TT-MALS solver (ALS with rank adaptation)."""

    def test_tt_mals_basic_call(self):
        """TT-MALS should run without errors."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        cores, info = tt_mals(
            x, y,
            memory_length=5,
            order=2,
            initial_ranks=[1, 2, 1]
        )

        assert len(cores) == 2
        assert 'final_ranks' in info
        assert 'rank_adaptation_steps' in info

    def test_tt_mals_with_config(self):
        """TT-MALS should accept config with rank adaptation params."""
        x = np.random.randn(100)
        y = np.random.randn(100)

        config = TTMALSConfig(
            max_iter=10,
            rank_adaptation=True,
            max_rank=5,
            rank_tol=1e-3,
            adapt_every=3
        )

        cores, info = tt_mals(
            x, y,
            memory_length=5,
            order=2,
            initial_ranks=[1, 2, 1],
            config=config
        )

        assert len(cores) == 2
        assert info['initial_ranks'] == [1, 2, 1]


class TestTTEdgeCases:
    """Test edge cases and special scenarios."""

    def test_small_memory_length(self):
        """N=2 (minimal memory) should work."""
        cores = [
            np.random.randn(1, 2, 2),
            np.random.randn(2, 2, 1),
        ]
        x = np.array([1.0, 0.5])

        y = tt_matvec(cores, x)
        assert isinstance(y, (float, np.floating))

    def test_large_rank(self):
        """Large rank should work."""
        N = 5
        cores = [
            np.random.randn(1, N, 10),
            np.random.randn(10, N, 1),
        ]
        x = np.random.randn(N)

        y = tt_matvec(cores, x)
        assert isinstance(y, (float, np.floating))

    def test_rank_one_full_tensor(self):
        """Rank-1 TT should give correct full tensor."""
        N = 3
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.5, 1.0, 1.5])

        cores = [
            a.reshape(1, N, 1),
            b.reshape(1, N, 1),
        ]

        A = tt_to_full(cores)
        A_expected = np.outer(a, b)

        assert_allclose(A, A_expected, rtol=1e-10)
