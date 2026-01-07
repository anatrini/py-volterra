"""
Comprehensive tests for tt_solvers_full module.

Tests full TT-ALS implementation with:
- Hankel delay matrix construction
- Tensor unfolding
- QR orthogonalization
- Least squares solving
- Full ALS algorithm
"""

import numpy as np
import pytest

from volterra.tt.tt_solvers_full import (
    build_delay_matrix,
    build_unfolded_data_matrix,
    evaluate_tt_volterra,
    qr_orthogonalize_core,
    solve_core_least_squares,
    tt_als_full,
)


class TestBuildDelayMatrix:
    """Test Hankel delay matrix construction."""

    def test_basic_delay_matrix(self):
        """Test basic delay matrix construction."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X = build_delay_matrix(x, memory_length=3)

        # Actual implementation returns [x(t-2), x(t-1), x(t)]
        expected = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])

        np.testing.assert_array_equal(X, expected)

    def test_delay_matrix_shape(self):
        """Test delay matrix has correct shape."""
        x = np.random.randn(100)
        memory_length = 10

        X = build_delay_matrix(x, memory_length)

        T_valid = 100 - 10 + 1
        assert X.shape == (T_valid, memory_length)

    def test_delay_matrix_various_lengths(self):
        """Test delay matrix for various signal lengths."""
        for T in [50, 100, 200]:
            for M in [3, 5, 10]:
                x = np.random.randn(T)
                X = build_delay_matrix(x, M)
                assert X.shape == (T - M + 1, M)

    def test_delay_matrix_signal_too_short(self):
        """Test error when signal shorter than memory."""
        x = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Signal length .* < memory length"):
            build_delay_matrix(x, memory_length=5)

    def test_delay_matrix_wrong_ndim(self):
        """Test error for non-1D input."""
        x = np.random.randn(10, 2)
        with pytest.raises(ValueError, match="Input must be 1D"):
            build_delay_matrix(x, memory_length=3)

    def test_delay_matrix_values(self):
        """Test delay matrix contains correct values."""
        x = np.arange(10.0)  # [0, 1, 2, ..., 9]
        X = build_delay_matrix(x, memory_length=4)

        # First row should be [0, 1, 2, 3]
        np.testing.assert_array_equal(X[0], [0, 1, 2, 3])

        # Last row should be [6, 7, 8, 9]
        np.testing.assert_array_equal(X[-1], [6, 7, 8, 9])


class TestBuildUnfoldedDataMatrix:
    """Test unfolded data matrix construction."""

    def test_first_core_no_orth(self):
        """Test unfolded matrix for first core without orthogonalization."""
        X_delay = np.random.randn(50, 10)
        order = 3

        Phi = build_unfolded_data_matrix(
            X_delay, order, core_idx=0, left_cores=None, right_cores=None
        )

        # For first core without orth, should return X_delay
        assert Phi.shape[0] == X_delay.shape[0]

    def test_last_core_no_orth(self):
        """Test unfolded matrix for last core."""
        X_delay = np.random.randn(50, 10)
        order = 3

        Phi = build_unfolded_data_matrix(
            X_delay, order, core_idx=order - 1, left_cores=None, right_cores=None
        )

        assert Phi.shape[0] == X_delay.shape[0]

    def test_middle_core_no_orth(self):
        """Test unfolded matrix for middle core."""
        X_delay = np.random.randn(50, 10)
        order = 4

        Phi = build_unfolded_data_matrix(
            X_delay, order, core_idx=1, left_cores=None, right_cores=None
        )

        assert Phi.shape[0] == X_delay.shape[0]

    def test_first_core_with_right_cores(self):
        """Test unfolded matrix for first core with right orthogonalized cores."""
        np.random.seed(42)
        T, N = 30, 5
        order = 3
        X_delay = np.random.randn(T, N)

        # Create dummy right cores (cores 1 and 2)
        right_cores = [
            np.random.randn(2, N, 2),  # Core 1: (r0=2, N=5, r1=2)
            np.random.randn(2, N, 1),  # Core 2: (r1=2, N=5, r2=1)
        ]

        Phi = build_unfolded_data_matrix(
            X_delay, order, core_idx=0, left_cores=None, right_cores=right_cores
        )

        # Shape should be (T, r_left * N * r_right) = (T, 1 * N * r_right)
        # where r_right comes from right_product contraction
        assert Phi.shape[0] == T
        assert Phi.shape[1] > 0  # Should have contracted right cores

    def test_last_core_with_left_cores(self):
        """Test unfolded matrix for last core with left orthogonalized cores."""
        np.random.seed(42)
        T, N = 30, 5
        order = 3
        X_delay = np.random.randn(T, N)

        # Create dummy left cores (cores 0 and 1)
        left_cores = [
            np.random.randn(1, N, 2),  # Core 0: (r0=1, N=5, r1=2)
            np.random.randn(2, N, 2),  # Core 1: (r1=2, N=5, r2=2)
        ]

        Phi = build_unfolded_data_matrix(
            X_delay, order, core_idx=order - 1, left_cores=left_cores, right_cores=None
        )

        # Shape should be (T, r_left * N * r_right) = (T, r_left * N * 1)
        assert Phi.shape[0] == T
        assert Phi.shape[1] > 0  # Should have contracted left cores

    def test_middle_core_with_both_sides(self):
        """Test unfolded matrix for middle core with both left and right cores."""
        np.random.seed(42)
        T, N = 30, 5
        order = 4
        core_idx = 1

        X_delay = np.random.randn(T, N)

        # Left core (core 0)
        left_cores = [np.random.randn(1, N, 2)]

        # Right cores (cores 2 and 3)
        right_cores = [
            np.random.randn(2, N, 2),  # Core 2
            np.random.randn(2, N, 1),  # Core 3
        ]

        Phi = build_unfolded_data_matrix(
            X_delay, order, core_idx=core_idx, left_cores=left_cores, right_cores=right_cores
        )

        # Should have shape (T, r_left * N * r_right)
        assert Phi.shape[0] == T
        assert Phi.shape[1] > 0


class TestSolveCoreLeastSquares:
    """Test least squares solving for TT cores."""

    def test_solve_core_basic(self):
        """Test basic least squares solve."""
        np.random.seed(42)

        T, N = 100, 10
        r_left, r_right = 1, 2

        # Create synthetic data
        Phi = np.random.randn(T, r_left * N * r_right)
        y = np.random.randn(T)

        core = solve_core_least_squares(Phi, y, (r_left, N, r_right))

        assert core.shape == (r_left, N, r_right)

    def test_solve_core_shape_mismatch(self):
        """Test error when Phi shape doesn't match core shape."""
        T, N = 50, 10
        Phi = np.random.randn(T, 15)  # Wrong number of columns
        y = np.random.randn(T)

        with pytest.raises(ValueError, match="Phi has .* columns, expected"):
            solve_core_least_squares(Phi, y, (1, 10, 2))

    def test_solve_core_with_regularization(self):
        """Test least squares with different regularization."""
        np.random.seed(42)

        T, N = 100, 10
        Phi = np.random.randn(T, N)
        y = np.random.randn(T)

        core1 = solve_core_least_squares(Phi, y, (1, N, 1), regularization=0)
        core2 = solve_core_least_squares(Phi, y, (1, N, 1), regularization=1e-3)

        # With regularization, solution should be different
        assert not np.allclose(core1, core2)

    def test_solve_core_various_shapes(self):
        """Test solve for various core shapes."""
        np.random.seed(42)
        T = 100

        shapes = [(1, 10, 1), (1, 10, 3), (2, 10, 3), (3, 10, 1)]

        for r_left, N, r_right in shapes:
            Phi = np.random.randn(T, r_left * N * r_right)
            y = np.random.randn(T)

            core = solve_core_least_squares(Phi, y, (r_left, N, r_right))
            assert core.shape == (r_left, N, r_right)


class TestQROrthogonalizeCore:
    """Test QR orthogonalization of TT cores."""

    def test_left_orthogonalization(self):
        """Test left QR orthogonalization."""
        np.random.seed(42)
        core = np.random.randn(2, 10, 3)

        Q, R = qr_orthogonalize_core(core, direction="left")

        # Q should be orthogonal: Q^T Q = I
        Q_mat = Q.reshape(-1, Q.shape[2])
        assert np.allclose(Q_mat.T @ Q_mat, np.eye(Q_mat.shape[1]), atol=1e-10)

        # Reconstruction: Q @ R should approximate original
        # Q (2, 10, r_new), R (r_new, 3)
        reconstructed = np.einsum("ijk,kl->ijl", Q, R)
        assert np.allclose(reconstructed, core, atol=1e-10)

    def test_right_orthogonalization(self):
        """Test right QR orthogonalization."""
        np.random.seed(42)
        core = np.random.randn(2, 10, 3)

        Q, R = qr_orthogonalize_core(core, direction="right")

        # Q should be orthogonal
        Q_mat = Q.reshape(Q.shape[0], -1).T
        assert np.allclose(Q_mat.T @ Q_mat, np.eye(Q_mat.shape[1]), atol=1e-10)

        # Reconstruction: R @ Q should approximate original
        # R (2, r_new), Q (r_new, 10, 3)
        reconstructed = np.einsum("ij,jkl->ikl", R, Q)
        assert np.allclose(reconstructed, core, atol=1e-10)

    def test_invalid_direction(self):
        """Test error for invalid direction."""
        core = np.random.randn(2, 10, 3)

        with pytest.raises(ValueError, match="direction must be"):
            qr_orthogonalize_core(core, direction="invalid")

    def test_various_core_shapes(self):
        """Test QR for various core shapes."""
        shapes = [(1, 10, 1), (2, 5, 3), (3, 20, 2), (1, 15, 5)]

        for r_left, N, r_right in shapes:
            core = np.random.randn(r_left, N, r_right)

            # Left orthogonalization
            Q_left, R_left = qr_orthogonalize_core(core, direction="left")
            assert Q_left.shape[0] == r_left
            assert Q_left.shape[1] == N

            # Right orthogonalization
            Q_right, R_right = qr_orthogonalize_core(core, direction="right")
            assert Q_right.shape[1] == N
            assert Q_right.shape[2] == r_right


class TestEvaluateTTVolterra:
    """Test TT-Volterra evaluation."""

    def test_evaluate_basic(self):
        """Test basic TT-Volterra evaluation."""
        np.random.seed(42)

        # Create simple cores
        cores = [
            np.random.randn(1, 10, 2),
            np.random.randn(2, 10, 1),
        ]

        X_delay = np.random.randn(50, 10)

        y_pred = evaluate_tt_volterra(cores, X_delay)

        assert y_pred.shape == (50,)
        assert np.all(np.isfinite(y_pred))

    def test_evaluate_various_orders(self):
        """Test evaluation for various orders."""
        np.random.seed(42)
        X_delay = np.random.randn(30, 10)

        for order in [2, 3, 4]:
            cores = []
            ranks = [1] + [2] * (order - 1) + [1]

            for k in range(order):
                core = np.random.randn(ranks[k], 10, ranks[k + 1])
                cores.append(core)

            y_pred = evaluate_tt_volterra(cores, X_delay)
            assert y_pred.shape == (30,)


class TestTTALSFull:
    """Test complete TT-ALS algorithm."""

    def test_tt_als_rank_validation(self):
        """Test rank validation."""
        x = np.random.randn(100)
        y = np.random.randn(100)

        # Wrong number of ranks
        with pytest.raises(ValueError, match="Need .* ranks"):
            tt_als_full(x, y, memory_length=5, order=3, ranks=[1, 2, 1])

        # Non-boundary ranks
        with pytest.raises(ValueError, match="Boundary ranks must be 1"):
            tt_als_full(x, y, memory_length=5, order=2, ranks=[2, 2, 1])

    def test_tt_als_input_length_mismatch(self):
        """Test error when x and y have different lengths."""
        x = np.random.randn(100)
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="x and y must have same length"):
            tt_als_full(x, y, memory_length=5, order=2, ranks=[1, 2, 1])

    def test_tt_als_simple_linear_system(self):
        """Test TT-ALS on simple linear system."""
        np.random.seed(42)

        # Generate simple linear system: y = sum(h * x[n-i])
        N = 5
        T = 100
        h_true = np.array([0.5, 0.3, 0.2, 0.1, 0.05])

        x = np.random.randn(T) * 0.1
        y = np.convolve(x, h_true, mode="full")[:T]

        # Fit with TT-ALS (order 2 to test multi-core optimization)
        cores, info = tt_als_full(
            x, y, memory_length=N, order=2, ranks=[1, 2, 1], max_iter=5, verbose=False
        )

        # Check outputs
        assert len(cores) == 2
        assert cores[0].shape == (1, N, 2)
        assert cores[1].shape == (2, N, 1)
        assert len(info["loss_history"]) > 0
        assert info["iterations"] <= 5
        assert info["final_loss"] < np.inf

    def test_tt_als_convergence_behavior(self):
        """Test that TT-ALS reduces loss over iterations."""
        np.random.seed(123)

        # Simple quadratic system
        N = 4
        T = 80

        x = np.random.randn(T) * 0.1
        y = 0.5 * x + 0.2 * x**2 + np.random.randn(T) * 0.01

        cores, info = tt_als_full(
            x, y, memory_length=N, order=2, ranks=[1, 2, 1], max_iter=10, verbose=False
        )

        # Loss should decrease
        loss_history = info["loss_history"]
        assert len(loss_history) > 1

        # Check that loss generally decreases (allow some fluctuation)
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        assert final_loss <= initial_loss  # Should improve or stay same

    def test_tt_als_higher_order(self):
        """Test TT-ALS with higher order."""
        np.random.seed(456)

        N = 3
        T = 60
        order = 3

        x = np.random.randn(T) * 0.1
        y = x + 0.1 * x**2 + 0.05 * x**3

        cores, info = tt_als_full(
            x,
            y,
            memory_length=N,
            order=order,
            ranks=[1, 2, 2, 1],
            max_iter=5,
            verbose=False,
        )

        # Check structure
        assert len(cores) == order
        assert cores[0].shape == (1, N, 2)
        assert cores[1].shape == (2, N, 2)
        assert cores[2].shape == (2, N, 1)

    def test_tt_als_with_regularization(self):
        """Test TT-ALS with regularization."""
        np.random.seed(789)

        N = 4
        T = 100  # Increase sample size to avoid singular matrices

        x = np.random.randn(T) * 0.1
        y = 0.8 * x + 0.1 * x**2 + np.random.randn(T) * 0.01

        # With higher regularization
        cores_reg1, info_reg1 = tt_als_full(
            x,
            y,
            memory_length=N,
            order=2,
            ranks=[1, 2, 1],
            max_iter=5,
            regularization=1e-2,
            verbose=False,
        )

        # With lower regularization
        cores_reg2, info_reg2 = tt_als_full(
            x,
            y,
            memory_length=N,
            order=2,
            ranks=[1, 2, 1],
            max_iter=5,
            regularization=1e-4,
            verbose=False,
        )

        # Solutions should differ due to different regularization
        assert not np.allclose(cores_reg1[0], cores_reg2[0])

    def test_tt_als_verbose_output(self, capsys):
        """Test that verbose mode produces output."""
        np.random.seed(999)

        N = 3
        T = 50

        x = np.random.randn(T) * 0.1
        y = x + 0.1 * x**2

        tt_als_full(
            x, y, memory_length=N, order=2, ranks=[1, 2, 1], max_iter=3, verbose=True
        )

        captured = capsys.readouterr()
        # Should print iteration info
        assert "Iteration" in captured.out or "loss" in captured.out

    def test_tt_als_early_convergence(self):
        """Test that TT-ALS can converge early."""
        np.random.seed(111)

        N = 3
        T = 50

        # Very simple system that should converge quickly
        x = np.random.randn(T) * 0.01
        y = x  # Pure linear

        cores, info = tt_als_full(
            x,
            y,
            memory_length=N,
            order=2,
            ranks=[1, 1, 1],  # Rank-1 (simplest)
            max_iter=20,
            tol=1e-4,
            verbose=False,
        )

        # Should converge before max_iter
        # (may or may not, depending on tolerance, but test the flag exists)
        assert "converged" in info
        assert isinstance(info["converged"], bool)

    def test_tt_als_prediction_consistency(self):
        """Test that ALS-fitted cores produce predictions via evaluate_tt_volterra."""
        np.random.seed(222)

        N = 4
        T = 60

        x = np.random.randn(T) * 0.1
        y = 0.5 * x + 0.1 * x**2

        cores, info = tt_als_full(
            x, y, memory_length=N, order=2, ranks=[1, 2, 1], max_iter=5, verbose=False
        )

        # Build delay matrix
        X_delay = build_delay_matrix(x, N)

        # Evaluate
        y_pred = evaluate_tt_volterra(cores, X_delay)

        # Should produce valid predictions
        assert y_pred.shape[0] > 0
        assert np.all(np.isfinite(y_pred))
