"""
Tests for Generalized Memory Polynomial (GMP) model.

The GMP model is intermediate between diagonal Memory Polynomial and full TT-Volterra.
It includes diagonal terms plus selective cross-memory terms.
"""

import numpy as np
import pytest
from volterra.models.gmp import (
    GeneralizedMemoryPolynomial,
    GMPConfig,
)


class TestGMPInitialization:
    """Test GMP model initialization and validation."""

    def test_init_minimal(self):
        """Initialize with minimal parameters."""
        model = GeneralizedMemoryPolynomial(memory_length=5, order=3)
        assert model.memory_length == 5
        assert model.order == 3
        assert not model.is_fitted
        assert model.config is not None

    def test_init_with_config(self):
        """Initialize with custom configuration."""
        config = GMPConfig(
            max_cross_lag_distance=2,
            max_cross_order=3,
            regularization=1e-6
        )
        model = GeneralizedMemoryPolynomial(
            memory_length=10,
            order=5,
            config=config
        )
        assert model.config.max_cross_lag_distance == 2
        assert model.config.max_cross_order == 3
        assert model.config.regularization == 1e-6

    def test_init_invalid_memory_length(self):
        """Reject invalid memory_length."""
        with pytest.raises(ValueError, match="memory_length must be positive"):
            GeneralizedMemoryPolynomial(memory_length=0, order=3)

        with pytest.raises(ValueError, match="memory_length must be positive"):
            GeneralizedMemoryPolynomial(memory_length=-5, order=3)

    def test_init_invalid_order(self):
        """Reject invalid order."""
        with pytest.raises(ValueError, match="order must be at least 1"):
            GeneralizedMemoryPolynomial(memory_length=5, order=0)

        with pytest.raises(ValueError, match="order must be at least 1"):
            GeneralizedMemoryPolynomial(memory_length=5, order=-2)

    def test_config_invalid_cross_lag_distance(self):
        """Reject invalid max_cross_lag_distance."""
        with pytest.raises(ValueError, match="max_cross_lag_distance must be non-negative"):
            GMPConfig(max_cross_lag_distance=-1)

    def test_config_invalid_cross_order(self):
        """Reject invalid max_cross_order."""
        with pytest.raises(ValueError, match="max_cross_order must be at least 1"):
            GMPConfig(max_cross_order=0)

    def test_total_terms_diagonal_only(self):
        """Count terms correctly for diagonal-only (no cross-terms)."""
        config = GMPConfig(max_cross_lag_distance=0)
        model = GeneralizedMemoryPolynomial(
            memory_length=5,
            order=3,
            config=config
        )
        # Diagonal terms: N * M = 5 * 3 = 15
        assert model.total_terms == 15

    def test_total_terms_with_cross_terms(self):
        """Count terms correctly with cross-terms enabled."""
        config = GMPConfig(
            max_cross_lag_distance=2,
            max_cross_order=2
        )
        model = GeneralizedMemoryPolynomial(
            memory_length=5,
            order=3,
            config=config
        )
        # Should have more terms than diagonal-only
        assert model.total_terms > 15


class TestGMPFitting:
    """Test GMP model fitting."""

    def test_fit_siso_diagonal_linear_system(self):
        """Fit SISO linear system with diagonal-only terms."""
        np.random.seed(42)
        N = 5
        h_true = np.random.randn(N)

        # Generate test data
        x = np.random.randn(500)
        y_valid = np.convolve(x, h_true, mode='valid')
        y = np.concatenate([np.zeros(N-1), y_valid])

        # Fit diagonal-only GMP (equivalent to MP)
        config = GMPConfig(max_cross_lag_distance=0)
        model = GeneralizedMemoryPolynomial(
            memory_length=N,
            order=1,
            config=config
        )
        model.fit(x, y)

        assert model.is_fitted
        y_pred = model.predict(x)

        # Should recover linear system accurately
        nmse = np.mean((y - y_pred) ** 2) / np.var(y)
        assert nmse < 1e-8

    def test_fit_siso_quadratic_system(self):
        """Fit SISO nonlinear system with quadratic terms."""
        np.random.seed(43)
        N = 5
        h1 = np.random.randn(N) * 0.5
        h2 = np.random.randn(N) * 0.1

        # Generate test data: y = h1 * x + h2 * x^2
        x = np.random.randn(500)
        from volterra.tt.tt_als_mimo import build_mimo_delay_matrix
        X_delay = build_mimo_delay_matrix(x, N)
        y_valid = X_delay @ h1 + (X_delay ** 2) @ h2
        y = np.concatenate([np.zeros(N-1), y_valid])

        # Fit diagonal GMP with order 2
        config = GMPConfig(max_cross_lag_distance=0)
        model = GeneralizedMemoryPolynomial(
            memory_length=N,
            order=2,
            config=config
        )
        model.fit(x, y)

        assert model.is_fitted
        y_pred = model.predict(x)

        # Should fit well
        nmse = np.mean((y - y_pred) ** 2) / np.var(y)
        assert nmse < 1e-6

    def test_fit_with_cross_terms_improves_accuracy(self):
        """Cross-terms should improve fit for systems with memory interactions."""
        np.random.seed(44)
        N = 5

        # Generate system with cross-memory interaction: y(t) = x(t) + 0.3*x(t-1)*x(t-2)
        x = np.random.randn(500)
        y = np.zeros_like(x)
        for t in range(2, len(x)):
            y[t] = x[t] + 0.3 * x[t-1] * x[t-2]

        # Fit diagonal-only model
        config_diag = GMPConfig(max_cross_lag_distance=0)
        model_diag = GeneralizedMemoryPolynomial(
            memory_length=N,
            order=2,
            config=config_diag
        )
        model_diag.fit(x, y)
        y_pred_diag = model_diag.predict(x)
        nmse_diag = np.mean((y - y_pred_diag) ** 2) / np.var(y)

        # Fit with cross-terms
        config_cross = GMPConfig(
            max_cross_lag_distance=3,
            max_cross_order=2
        )
        model_cross = GeneralizedMemoryPolynomial(
            memory_length=N,
            order=2,
            config=config_cross
        )
        model_cross.fit(x, y)
        y_pred_cross = model_cross.predict(x)
        nmse_cross = np.mean((y - y_pred_cross) ** 2) / np.var(y)

        # Cross-term model should fit better
        assert nmse_cross < nmse_diag
        assert nmse_cross < 0.01  # Reasonable fit, but regularization prevents perfect recovery

    def test_fit_mimo_two_inputs(self):
        """Fit MIMO system with two input channels."""
        np.random.seed(45)
        N = 5
        h1 = np.random.randn(N) * 0.5
        h2 = np.random.randn(N) * 0.3

        # Generate MIMO data
        x = np.random.randn(500, 2)
        from volterra.tt.tt_als_mimo import build_mimo_delay_matrix
        X1_delay = build_mimo_delay_matrix(x[:, 0], N)
        X2_delay = build_mimo_delay_matrix(x[:, 1], N)
        y_valid = X1_delay @ h1 + X2_delay @ h2
        y = np.concatenate([np.zeros(N-1), y_valid])

        # Fit GMP
        config = GMPConfig(max_cross_lag_distance=0)
        model = GeneralizedMemoryPolynomial(
            memory_length=N,
            order=1,
            config=config
        )
        model.fit(x, y)

        assert model.is_fitted
        y_pred = model.predict(x)

        nmse = np.mean((y - y_pred) ** 2) / np.var(y)
        assert nmse < 1e-6

    def test_fit_multi_output(self):
        """Fit system with multiple outputs."""
        np.random.seed(46)
        N = 5

        x = np.random.randn(200)
        from volterra.tt.tt_als_mimo import build_mimo_delay_matrix
        X_delay = build_mimo_delay_matrix(x, N)

        h1 = np.random.randn(N) * 0.5
        h2 = np.random.randn(N) * 0.3
        y1_valid = X_delay @ h1
        y2_valid = X_delay @ h2
        y = np.column_stack([
            np.concatenate([np.zeros(N-1), y1_valid]),
            np.concatenate([np.zeros(N-1), y2_valid])
        ])

        config = GMPConfig(max_cross_lag_distance=0)
        model = GeneralizedMemoryPolynomial(
            memory_length=N,
            order=1,
            config=config
        )
        model.fit(x, y)

        assert model.is_fitted
        assert model.n_outputs == 2

    def test_fit_sets_fitted_flag(self):
        """Fitting should set is_fitted flag."""
        np.random.seed(47)
        x = np.random.randn(100)
        y = np.random.randn(100)

        model = GeneralizedMemoryPolynomial(memory_length=5, order=2)
        assert not model.is_fitted

        model.fit(x, y)
        assert model.is_fitted


class TestGMPPrediction:
    """Test GMP model prediction."""

    def test_predict_before_fit_raises(self):
        """Predicting before fitting should raise error."""
        model = GeneralizedMemoryPolynomial(memory_length=5, order=2)
        x = np.random.randn(100)

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.predict(x)

    def test_predict_siso_returns_correct_shape(self):
        """Prediction should return correct shape for SISO."""
        np.random.seed(48)
        x_train = np.random.randn(200)
        y_train = np.random.randn(200)

        model = GeneralizedMemoryPolynomial(memory_length=5, order=2)
        model.fit(x_train, y_train)

        x_test = np.random.randn(100)
        y_pred = model.predict(x_test)

        assert y_pred.shape == (100,)

    def test_predict_mimo_returns_correct_shape(self):
        """Prediction should return correct shape for MIMO."""
        np.random.seed(49)
        x_train = np.random.randn(200, 3)
        y_train = np.random.randn(200)

        model = GeneralizedMemoryPolynomial(memory_length=5, order=2)
        model.fit(x_train, y_train)

        x_test = np.random.randn(150, 3)
        y_pred = model.predict(x_test)

        assert y_pred.shape == (150,)

    def test_predict_multi_output_returns_correct_shape(self):
        """Prediction should return correct shape for multi-output."""
        np.random.seed(50)
        x_train = np.random.randn(200)
        y_train = np.random.randn(200, 2)

        model = GeneralizedMemoryPolynomial(memory_length=5, order=2)
        model.fit(x_train, y_train)

        x_test = np.random.randn(100)
        y_pred = model.predict(x_test)

        assert y_pred.shape == (100, 2)

    def test_predict_deterministic(self):
        """Prediction should be deterministic."""
        np.random.seed(51)
        x_train = np.random.randn(200)
        y_train = np.random.randn(200)

        model = GeneralizedMemoryPolynomial(memory_length=5, order=2)
        model.fit(x_train, y_train)

        x_test = np.random.randn(100)
        y_pred1 = model.predict(x_test)
        y_pred2 = model.predict(x_test)

        np.testing.assert_allclose(y_pred1, y_pred2)

    def test_predict_different_length_input(self):
        """Prediction should work with different length input than training."""
        np.random.seed(52)
        x_train = np.random.randn(500)
        y_train = np.random.randn(500)

        model = GeneralizedMemoryPolynomial(memory_length=5, order=2)
        model.fit(x_train, y_train)

        # Test with shorter input
        x_short = np.random.randn(50)
        y_short = model.predict(x_short)
        assert y_short.shape == (50,)

        # Test with longer input
        x_long = np.random.randn(1000)
        y_long = model.predict(x_long)
        assert y_long.shape == (1000,)


class TestGMPAccessors:
    """Test GMP model accessors and export."""

    def test_get_coefficients_before_fit_raises(self):
        """Getting coefficients before fit should raise."""
        model = GeneralizedMemoryPolynomial(memory_length=5, order=2)

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.get_coefficients()

    def test_get_coefficients_structure(self):
        """Coefficients should be organized by term type."""
        np.random.seed(53)
        x = np.random.randn(200)
        y = np.random.randn(200)

        config = GMPConfig(max_cross_lag_distance=2, max_cross_order=2)
        model = GeneralizedMemoryPolynomial(
            memory_length=5,
            order=2,
            config=config
        )
        model.fit(x, y)

        coeffs = model.get_coefficients()

        assert 'diagonal' in coeffs
        assert isinstance(coeffs['diagonal'], np.ndarray)

        # If cross-terms enabled, should have cross-term coefficients
        if config.max_cross_lag_distance > 0:
            assert 'cross_terms' in coeffs
            assert isinstance(coeffs['cross_terms'], list) or isinstance(coeffs['cross_terms'], np.ndarray)

    def test_get_coefficients_multi_output(self):
        """Get coefficients for specific output channel."""
        np.random.seed(54)
        x = np.random.randn(200)
        y = np.random.randn(200, 2)

        model = GeneralizedMemoryPolynomial(memory_length=5, order=2)
        model.fit(x, y)

        coeffs_0 = model.get_coefficients(output_idx=0)
        coeffs_1 = model.get_coefficients(output_idx=1)

        # Should be different for different outputs
        assert not np.allclose(coeffs_0['diagonal'], coeffs_1['diagonal'])

    def test_export_model_structure(self):
        """Export should provide complete model description."""
        np.random.seed(55)
        x = np.random.randn(200)
        y = np.random.randn(200)

        config = GMPConfig(max_cross_lag_distance=1)
        model = GeneralizedMemoryPolynomial(
            memory_length=5,
            order=3,
            config=config
        )
        model.fit(x, y)

        export = model.export_model()

        # Check required fields
        assert 'model_type' in export
        assert export['model_type'] == 'GMP'
        assert 'memory_length' in export
        assert 'order' in export
        assert 'coefficients' in export
        assert 'config' in export
        assert export['memory_length'] == 5
        assert export['order'] == 3

    def test_export_model_invalid_output_idx(self):
        """Export with invalid output index should raise."""
        np.random.seed(56)
        x = np.random.randn(200)
        y = np.random.randn(200)

        model = GeneralizedMemoryPolynomial(memory_length=5, order=2)
        model.fit(x, y)

        with pytest.raises(IndexError, match="output_idx"):
            model.export_model(output_idx=5)


class TestGMPProperties:
    """Test GMP model properties."""

    def test_repr(self):
        """String representation should be informative."""
        config = GMPConfig(max_cross_lag_distance=2)
        model = GeneralizedMemoryPolynomial(
            memory_length=10,
            order=3,
            config=config
        )

        repr_str = repr(model)
        assert 'GeneralizedMemoryPolynomial' in repr_str
        assert 'memory_length=10' in repr_str
        assert 'order=3' in repr_str
        assert 'fitted=False' in repr_str

    def test_repr_after_fit(self):
        """String representation should update after fit."""
        np.random.seed(57)
        x = np.random.randn(200)
        y = np.random.randn(200)

        model = GeneralizedMemoryPolynomial(memory_length=5, order=2)
        model.fit(x, y)

        repr_str = repr(model)
        assert 'fitted=True' in repr_str

    def test_n_inputs_property(self):
        """n_inputs should reflect input dimensionality."""
        np.random.seed(58)

        # SISO
        x_siso = np.random.randn(200)
        y_siso = np.random.randn(200)
        model_siso = GeneralizedMemoryPolynomial(memory_length=5, order=2)
        model_siso.fit(x_siso, y_siso)
        assert model_siso.n_inputs == 1

        # MIMO
        x_mimo = np.random.randn(200, 3)
        y_mimo = np.random.randn(200)
        model_mimo = GeneralizedMemoryPolynomial(memory_length=5, order=2)
        model_mimo.fit(x_mimo, y_mimo)
        assert model_mimo.n_inputs == 3

    def test_n_outputs_property(self):
        """n_outputs should reflect output dimensionality."""
        np.random.seed(59)
        x = np.random.randn(200)

        # Single output
        y_single = np.random.randn(200)
        model_single = GeneralizedMemoryPolynomial(memory_length=5, order=2)
        model_single.fit(x, y_single)
        assert model_single.n_outputs == 1

        # Multi output
        y_multi = np.random.randn(200, 4)
        model_multi = GeneralizedMemoryPolynomial(memory_length=5, order=2)
        model_multi.fit(x, y_multi)
        assert model_multi.n_outputs == 4

    def test_total_terms_property(self):
        """total_terms should count all terms correctly."""
        config = GMPConfig(max_cross_lag_distance=0)
        model = GeneralizedMemoryPolynomial(
            memory_length=5,
            order=3,
            config=config
        )

        # Diagonal only: N * M = 15
        assert model.total_terms == 15

    def test_is_diagonal_property(self):
        """is_diagonal should indicate if only diagonal terms present."""
        config_diag = GMPConfig(max_cross_lag_distance=0)
        model_diag = GeneralizedMemoryPolynomial(
            memory_length=5,
            order=3,
            config=config_diag
        )
        assert model_diag.is_diagonal

        config_cross = GMPConfig(max_cross_lag_distance=2)
        model_cross = GeneralizedMemoryPolynomial(
            memory_length=5,
            order=3,
            config=config_cross
        )
        assert not model_cross.is_diagonal


class TestGMPIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow_siso(self):
        """Complete workflow: init → fit → predict → export."""
        np.random.seed(60)

        # Generate synthetic nonlinear system
        N = 10
        M = 3
        x_train = np.random.randn(1000)

        # True system: diagonal + one cross-term
        from volterra.tt.tt_als_mimo import build_mimo_delay_matrix
        X_delay = build_mimo_delay_matrix(x_train, N)
        h1 = np.random.randn(N) * 0.5
        h2 = np.random.randn(N) * 0.1
        h3 = np.random.randn(N) * 0.03
        y_valid = X_delay @ h1 + (X_delay ** 2) @ h2 + (X_delay ** 3) @ h3

        # Add cross-term: x(t-1) * x(t-3)
        for t in range(3, len(X_delay)):
            y_valid[t] += 0.2 * X_delay[t, 1] * X_delay[t, 3]

        y_train = np.concatenate([np.zeros(N-1), y_valid])

        # Fit GMP with cross-terms
        config = GMPConfig(
            max_cross_lag_distance=3,
            max_cross_order=2,
            regularization=1e-6
        )
        model = GeneralizedMemoryPolynomial(
            memory_length=N,
            order=M,
            config=config
        )
        model.fit(x_train, y_train)

        # Predict on new data
        x_test = np.random.randn(500)
        y_test_pred = model.predict(x_test)
        assert y_test_pred.shape == (500,)

        # Export model
        export = model.export_model()
        assert export['model_type'] == 'GMP'
        assert export['memory_length'] == N
        assert export['order'] == M

    def test_complete_workflow_mimo(self):
        """Complete workflow for MIMO system."""
        np.random.seed(61)

        # Generate MIMO system
        N = 8
        M = 2
        I = 3  # 3 inputs

        x_train = np.random.randn(800, I)

        # Simple additive MIMO model
        from volterra.tt.tt_als_mimo import build_mimo_delay_matrix
        y_valid = np.zeros(800 - N + 1)
        for i in range(I):
            X_delay = build_mimo_delay_matrix(x_train[:, i], N)
            h = np.random.randn(N) * 0.3
            y_valid += X_delay @ h

        y_train = np.concatenate([np.zeros(N-1), y_valid])

        # Fit GMP
        config = GMPConfig(max_cross_lag_distance=0)  # Diagonal only for simplicity
        model = GeneralizedMemoryPolynomial(
            memory_length=N,
            order=M,
            config=config
        )
        model.fit(x_train, y_train)

        # Predict
        x_test = np.random.randn(400, I)
        y_test_pred = model.predict(x_test)
        assert y_test_pred.shape == (400,)

        # Check properties
        assert model.n_inputs == I
        assert model.n_outputs == 1
        assert model.is_fitted
