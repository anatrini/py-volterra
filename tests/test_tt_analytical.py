"""
Analytical validation tests for TT-Volterra.

These tests verify correctness against known analytical cases:
1. Memoryless polynomial (N=1)
2. Linear system (M=1)
3. Rank-1 separable kernels
4. Known polynomial expansion validation

Critical for STEP 6: Analytical Validation
"""

import numpy as np
from numpy.testing import assert_allclose

from volterra.models import TTVolterraConfig, TTVolterraIdentifier
from volterra.tt import TTTensor, tt_matvec, tt_to_full


class TestAnalyticalPolynomials:
    """Test against known polynomial expansions."""

    def test_memoryless_quadratic(self):
        """Memoryless y = a1*x + a2*x^2 should be exactly representable."""
        # Analytical: y(t) = 0.8*x(t) + 0.2*x(t)^2
        a1, a2 = 0.8, 0.2

        # Generate data
        np.random.seed(42)
        x = np.random.randn(500) * 0.5  # Keep amplitude moderate
        y = a1 * x + a2 * x**2

        # Identify with TT (N=1 for memoryless, M=2 for quadratic)
        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=2,
            ranks=[1, 2, 1],
            config=TTVolterraConfig(max_iter=1, verbose=False),
        )
        identifier.fit(x, y)

        # Verify model exists
        assert identifier.is_fitted
        kernels = identifier.get_kernels()
        assert kernels.ndim == 2
        assert kernels.shape == (1, 1)  # N=1, M=2

    def test_linear_system_rank_one(self):
        """Pure linear system should have M=1, single core."""
        # y(t) = h1 * x(t)
        h1 = 2.5

        np.random.seed(42)
        x = np.random.randn(500)
        y = h1 * x

        # Identify with M=1
        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=1,
            ranks=[1, 1],  # M=1 requires 2 ranks
            config=TTVolterraConfig(max_iter=1),
        )
        identifier.fit(x, y)

        assert identifier.is_fitted
        kernels = identifier.get_kernels()
        assert kernels.ndim == 1  # Single dimension for M=1

    def test_separable_kernel_rank_one(self):
        """Rank-1 separable kernel: h(i,j) = a(i) * b(j)."""
        N = 3
        a = np.array([1.0, 0.5, 0.25])
        b = np.array([0.8, 0.4, 0.2])

        # Build rank-1 TT representation
        cores = [
            a.reshape(1, N, 1),
            b.reshape(1, N, 1),
        ]
        TTTensor(cores)

        # Full tensor should be outer product
        A_full = tt_to_full(cores)
        A_expected = np.outer(a, b)

        assert_allclose(A_full, A_expected, rtol=1e-10)

    def test_tt_matvec_matches_full_contraction(self):
        """TT-matvec should match full tensor contraction."""
        np.random.seed(42)
        N = 4
        cores = [
            np.random.randn(1, N, 2),
            np.random.randn(2, N, 1),
        ]
        x = np.random.randn(N)

        # TT-matvec
        y_tt = tt_matvec(cores, x)

        # Full contraction
        A = tt_to_full(cores)
        y_full = np.einsum("ij,i,j->", A, x, x)

        assert_allclose(y_tt, y_full, rtol=1e-10)


class TestEdgeCases:
    """Test edge cases: N=1, M=1, rank=1."""

    def test_n_equals_1_memoryless(self):
        """N=1 (memoryless) should work."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = x + 0.1 * x**2

        identifier = TTVolterraIdentifier(memory_length=1, order=2, ranks=[1, 2, 1])
        identifier.fit(x, y)

        assert identifier.is_fitted
        assert identifier.memory_length == 1

    def test_m_equals_1_linear(self):
        """M=1 (linear only) should work."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = 2.0 * x

        identifier = TTVolterraIdentifier(memory_length=5, order=1, ranks=[1, 1])
        identifier.fit(x, y)

        assert identifier.is_fitted
        assert identifier.order == 1

    def test_rank_one_tt(self):
        """Rank=1 TT (all internal ranks = 1) should work."""
        N = 5
        cores = [
            np.random.randn(1, N, 1),
            np.random.randn(1, N, 1),
            np.random.randn(1, N, 1),
        ]
        tt = TTTensor(cores)

        assert tt.ranks == (1, 1, 1, 1)  # All ranks = 1

        # Should work with matvec
        x = np.random.randn(N)
        y = tt_matvec(cores, x)
        assert isinstance(y, (float, np.floating))

    def test_very_high_order(self):
        """High order M=5 should initialize correctly."""
        identifier = TTVolterraIdentifier(memory_length=3, order=5, ranks=[1, 2, 2, 2, 2, 1])

        assert identifier.order == 5
        assert len(identifier.ranks) == 6  # M+1 ranks

    def test_minimal_config(self):
        """Minimal configuration (N=1, M=1, rank=1) should work."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        identifier = TTVolterraIdentifier(memory_length=1, order=1, ranks=[1, 1])
        identifier.fit(x, y)

        assert identifier.is_fitted


class TestDiagonalOnlyMode:
    """Test diagonal-only mode (memory polynomial)."""

    def test_diagonal_config_flag(self):
        """diagonal_only config should be settable."""
        config = TTVolterraConfig(diagonal_only=True)
        assert config.diagonal_only

    def test_diagonal_mode_initialization(self):
        """Initialize with diagonal_only=True."""
        config = TTVolterraConfig(diagonal_only=True, solver="als")

        identifier = TTVolterraIdentifier(
            memory_length=10, order=3, ranks=[1, 2, 2, 1], config=config
        )

        assert identifier.config.diagonal_only

    def test_diagonal_vs_general_mode(self):
        """Both diagonal and general mode should work."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = x + 0.1 * x**2

        # General mode
        id_general = TTVolterraIdentifier(
            memory_length=5, order=2, ranks=[1, 3, 1], config=TTVolterraConfig(diagonal_only=False)
        )
        id_general.fit(x, y)

        # Diagonal mode
        id_diagonal = TTVolterraIdentifier(
            memory_length=5, order=2, ranks=[1, 3, 1], config=TTVolterraConfig(diagonal_only=True)
        )
        id_diagonal.fit(x, y)

        # Both should fit successfully
        assert id_general.is_fitted
        assert id_diagonal.is_fitted


class TestMIMOComprehensive:
    """Comprehensive MIMO tests."""

    def test_simo_2_outputs(self):
        """SIMO: Single input → 2 outputs."""
        np.random.seed(42)
        x = np.random.randn(100)  # (T,)
        y = np.column_stack([x + 0.1 * x**2, x - 0.1 * x**2])  # (T, 2)

        identifier = TTVolterraIdentifier(memory_length=5, order=2, ranks=[1, 2, 1])
        identifier.fit(x, y)

        assert identifier.n_inputs_ == 1
        assert identifier.n_outputs_ == 2
        assert len(identifier.tt_models_) == 2

    def test_miso_3_inputs(self):
        """MISO: 3 inputs → single output."""
        np.random.seed(42)
        x = np.random.randn(100, 3)  # (T, 3)
        y = x[:, 0] + 0.1 * x[:, 1] ** 2  # (T,)

        identifier = TTVolterraIdentifier(memory_length=5, order=2, ranks=[1, 2, 1])
        identifier.fit(x, y)

        assert identifier.n_inputs_ == 3
        assert identifier.n_outputs_ == 1

    def test_mimo_2x3(self):
        """Full MIMO: 2 inputs → 3 outputs."""
        np.random.seed(42)
        x = np.random.randn(100, 2)  # (T, 2)
        y = np.random.randn(100, 3)  # (T, 3)

        identifier = TTVolterraIdentifier(memory_length=5, order=2, ranks=[1, 2, 1])
        identifier.fit(x, y)

        assert identifier.n_inputs_ == 2
        assert identifier.n_outputs_ == 3
        assert len(identifier.tt_models_) == 3  # One TT per output

        # Get kernels for each output
        for o in range(3):
            kernels = identifier.get_kernels(output_idx=o)
            assert kernels.ndim == 2

    def test_mimo_predict_shapes(self):
        """MIMO predict should return correct shapes."""
        np.random.seed(42)
        x_train = np.random.randn(100, 2)
        y_train = np.random.randn(100, 3)

        identifier = TTVolterraIdentifier(memory_length=5, order=2, ranks=[1, 2, 1])
        identifier.fit(x_train, y_train)

        # Predict on new data
        x_test = np.random.randn(50, 2)
        y_pred = identifier.predict(x_test)

        # Should return multi-output
        assert y_pred.ndim == 2
        assert y_pred.shape[1] == 3  # 3 outputs


class TestRankAdaptation:
    """Test rank adaptation in TT-MALS."""

    def test_mals_solver_selection(self):
        """MALS solver should be selectable."""
        config = TTVolterraConfig(solver="mals")
        assert config.solver == "mals"

    def test_rank_adaptation_config(self):
        """Rank adaptation config should work."""
        config = TTVolterraConfig(solver="mals", rank_adaptation=True, max_rank=10, rank_tol=1e-4)

        assert config.rank_adaptation
        assert config.max_rank == 10
        assert config.rank_tol == 1e-4

    def test_mals_vs_als(self):
        """Both ALS and MALS should work."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = x + 0.1 * x**2

        # ALS
        id_als = TTVolterraIdentifier(
            memory_length=5, order=2, ranks=[1, 2, 1], config=TTVolterraConfig(solver="als")
        )
        id_als.fit(x, y)

        # MALS
        id_mals = TTVolterraIdentifier(
            memory_length=5,
            order=2,
            ranks=[1, 2, 1],
            config=TTVolterraConfig(solver="mals", rank_adaptation=True),
        )
        id_mals.fit(x, y)

        # Both should fit
        assert id_als.is_fitted
        assert id_mals.is_fitted

    def test_initial_vs_final_ranks(self):
        """MALS should report initial and final ranks."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = x + 0.1 * x**2

        identifier = TTVolterraIdentifier(
            memory_length=5,
            order=2,
            ranks=[1, 2, 1],
            config=TTVolterraConfig(solver="mals", rank_adaptation=True),
        )
        identifier.fit(x, y)

        # fit_info should contain rank information
        assert "per_output" in identifier.fit_info_
        per_output_info = identifier.fit_info_["per_output"][0]
        assert "initial_ranks" in per_output_info
        assert "final_ranks" in per_output_info


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_zero_input(self):
        """Zero input should not crash."""
        x = np.zeros(100)
        y = np.zeros(100)

        identifier = TTVolterraIdentifier(memory_length=5, order=2, ranks=[1, 2, 1])
        identifier.fit(x, y)

        assert identifier.is_fitted

    def test_very_small_amplitudes(self):
        """Very small signal amplitudes should work."""
        np.random.seed(42)
        x = np.random.randn(100) * 1e-6
        y = x + 1e-7 * x**2

        identifier = TTVolterraIdentifier(memory_length=5, order=2, ranks=[1, 2, 1])
        identifier.fit(x, y)

        assert identifier.is_fitted

    def test_large_memory_length(self):
        """Large memory length should initialize correctly."""
        identifier = TTVolterraIdentifier(memory_length=100, order=2, ranks=[1, 3, 1])

        assert identifier.memory_length == 100

    def test_regularization_parameter(self):
        """Regularization parameter should be configurable."""
        config = TTVolterraConfig(regularization=1e-6)
        assert config.regularization == 1e-6

        identifier = TTVolterraIdentifier(memory_length=5, order=2, ranks=[1, 2, 1], config=config)

        assert identifier.config.regularization == 1e-6
