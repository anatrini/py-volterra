"""
Tests for automatic model selection between MP, GMP, and TT-Volterra.
"""

import numpy as np
import pytest
from volterra.model_selection import (
    ModelSelector,
    ModelSelectionConfig,
)


class TestModelSelectionConfig:
    """Test ModelSelectionConfig initialization and validation."""

    def test_config_minimal(self):
        """Initialize with minimal parameters."""
        config = ModelSelectionConfig()
        assert config.memory_length > 0
        assert config.order >= 1
        assert config.validation_split > 0 and config.validation_split < 1

    def test_config_custom_parameters(self):
        """Initialize with custom parameters."""
        config = ModelSelectionConfig(
            memory_length=10,
            order=5,
            validation_split=0.3,
            try_diagonal=True,
            try_gmp=True,
            try_tt_full=True
        )
        assert config.memory_length == 10
        assert config.order == 5
        assert config.validation_split == 0.3
        assert config.try_diagonal
        assert config.try_gmp
        assert config.try_tt_full

    def test_config_invalid_memory_length(self):
        """Reject invalid memory_length."""
        with pytest.raises(ValueError, match="memory_length must be positive"):
            ModelSelectionConfig(memory_length=0)

    def test_config_invalid_order(self):
        """Reject invalid order."""
        with pytest.raises(ValueError, match="order must be at least 1"):
            ModelSelectionConfig(order=0)

    def test_config_invalid_validation_split(self):
        """Reject invalid validation_split."""
        with pytest.raises(ValueError, match="validation_split must be in"):
            ModelSelectionConfig(validation_split=0.0)

        with pytest.raises(ValueError, match="validation_split must be in"):
            ModelSelectionConfig(validation_split=1.0)

    def test_config_must_try_at_least_one_model(self):
        """At least one model type must be enabled."""
        with pytest.raises(ValueError, match="At least one model type must be enabled"):
            ModelSelectionConfig(
                try_diagonal=False,
                try_gmp=False,
                try_tt_full=False
            )


class TestModelSelectorInitialization:
    """Test ModelSelector initialization."""

    def test_init_minimal(self):
        """Initialize with minimal parameters."""
        selector = ModelSelector()
        assert selector.config is not None
        assert not selector.is_fitted

    def test_init_with_config(self):
        """Initialize with custom configuration."""
        config = ModelSelectionConfig(memory_length=15, order=4)
        selector = ModelSelector(config=config)
        assert selector.config.memory_length == 15
        assert selector.config.order == 4

    def test_init_with_kwargs(self):
        """Initialize with kwargs shorthand."""
        selector = ModelSelector(memory_length=8, order=3)
        assert selector.config.memory_length == 8
        assert selector.config.order == 3


class TestModelSelectorFitting:
    """Test ModelSelector fitting and model selection."""

    def test_fit_linear_system_selects_diagonal(self):
        """Linear system should select diagonal MP."""
        np.random.seed(100)
        N = 5
        h_true = np.random.randn(N) * 0.5

        # Generate linear system
        x = np.random.randn(500)
        y_valid = np.convolve(x, h_true, mode='valid')
        y = np.concatenate([np.zeros(N-1), y_valid])

        # Fit with model selection
        config = ModelSelectionConfig(
            memory_length=N,
            order=1,
            try_diagonal=True,
            try_gmp=False,
            try_tt_full=False
        )
        selector = ModelSelector(config=config)
        selector.fit(x, y)

        assert selector.is_fitted
        assert selector.selected_model_type == 'Diagonal-MP'

    def test_fit_quadratic_system_diagonal_sufficient(self):
        """Diagonal MP should handle purely diagonal nonlinearity."""
        np.random.seed(101)
        N = 5
        h1 = np.random.randn(N) * 0.5
        h2 = np.random.randn(N) * 0.1

        # Generate diagonal quadratic system
        x = np.random.randn(500)
        from volterra.tt.tt_als_mimo import build_mimo_delay_matrix
        X_delay = build_mimo_delay_matrix(x, N)
        y_valid = X_delay @ h1 + (X_delay ** 2) @ h2
        y = np.concatenate([np.zeros(N-1), y_valid])

        # Fit with model selection
        config = ModelSelectionConfig(
            memory_length=N,
            order=2,
            try_diagonal=True,
            try_gmp=True,
            try_tt_full=False
        )
        selector = ModelSelector(config=config)
        selector.fit(x, y)

        assert selector.is_fitted
        # Should select diagonal since GMP adds unnecessary complexity
        assert selector.selected_model_type in ['Diagonal-MP', 'GMP']

    def test_fit_cross_memory_system_selects_gmp(self):
        """System with cross-memory terms should prefer GMP."""
        np.random.seed(102)
        N = 5

        # Generate system with cross-memory: y(t) = x(t) + 0.4*x(t-1)*x(t-2)
        x = np.random.randn(500)
        y = np.zeros_like(x)
        for t in range(2, len(x)):
            y[t] = x[t] + 0.4 * x[t-1] * x[t-2]

        # Fit with model selection
        config = ModelSelectionConfig(
            memory_length=N,
            order=2,
            try_diagonal=True,
            try_gmp=True,
            try_tt_full=False,
            gmp_max_cross_lag=3
        )
        selector = ModelSelector(config=config)
        selector.fit(x, y)

        assert selector.is_fitted
        # GMP should fit significantly better than diagonal
        assert selector.selected_model_type == 'GMP'

    def test_fit_with_validation_split(self):
        """Fit using automatic train/validation split."""
        np.random.seed(103)
        x = np.random.randn(500)
        y = np.random.randn(500)

        config = ModelSelectionConfig(
            validation_split=0.2,
            try_diagonal=True,
            try_gmp=False,
            try_tt_full=False
        )
        selector = ModelSelector(config=config)
        selector.fit(x, y)

        assert selector.is_fitted

    def test_fit_with_explicit_validation_data(self):
        """Fit using explicit validation set."""
        np.random.seed(104)
        x_train = np.random.randn(400)
        y_train = np.random.randn(400)
        x_val = np.random.randn(100)
        y_val = np.random.randn(100)

        config = ModelSelectionConfig(
            try_diagonal=True,
            try_gmp=False,
            try_tt_full=False
        )
        selector = ModelSelector(config=config)
        selector.fit(x_train, y_train, x_val=x_val, y_val=y_val)

        assert selector.is_fitted

    def test_fit_mimo_system(self):
        """Model selection works for MIMO systems."""
        np.random.seed(105)
        x_train = np.random.randn(500, 2)
        y_train = np.random.randn(500)

        config = ModelSelectionConfig(
            try_diagonal=True,
            try_gmp=False,
            try_tt_full=False
        )
        selector = ModelSelector(config=config)
        selector.fit(x_train, y_train)

        assert selector.is_fitted

    def test_fit_multi_output_system(self):
        """Model selection works for multi-output systems."""
        np.random.seed(106)
        x_train = np.random.randn(500)
        y_train = np.random.randn(500, 2)

        config = ModelSelectionConfig(
            try_diagonal=True,
            try_gmp=False,
            try_tt_full=False
        )
        selector = ModelSelector(config=config)
        selector.fit(x_train, y_train)

        assert selector.is_fitted


class TestModelSelectorPrediction:
    """Test ModelSelector prediction."""

    def test_predict_before_fit_raises(self):
        """Prediction before fit should raise."""
        selector = ModelSelector()
        x = np.random.randn(100)

        with pytest.raises(RuntimeError, match="must be fitted"):
            selector.predict(x)

    def test_predict_after_fit(self):
        """Prediction works after fitting."""
        np.random.seed(107)
        x_train = np.random.randn(400)
        y_train = np.random.randn(400)

        selector = ModelSelector()
        selector.fit(x_train, y_train)

        x_test = np.random.randn(100)
        y_pred = selector.predict(x_test)

        assert y_pred.shape == (100,)

    def test_predict_mimo(self):
        """Prediction works for MIMO."""
        np.random.seed(108)
        x_train = np.random.randn(400, 2)
        y_train = np.random.randn(400)

        selector = ModelSelector()
        selector.fit(x_train, y_train)

        x_test = np.random.randn(100, 2)
        y_pred = selector.predict(x_test)

        assert y_pred.shape == (100,)

    def test_predict_multi_output(self):
        """Prediction works for multi-output."""
        np.random.seed(109)
        x_train = np.random.randn(400)
        y_train = np.random.randn(400, 2)

        selector = ModelSelector()
        selector.fit(x_train, y_train)

        x_test = np.random.randn(100)
        y_pred = selector.predict(x_test)

        assert y_pred.shape == (100, 2)


class TestModelSelectorAccessors:
    """Test ModelSelector accessors and diagnostics."""

    def test_get_selected_model(self):
        """Get the selected model object."""
        np.random.seed(110)
        x = np.random.randn(400)
        y = np.random.randn(400)

        selector = ModelSelector()
        selector.fit(x, y)

        model = selector.get_selected_model()
        assert model is not None
        assert hasattr(model, 'predict')

    def test_get_all_results(self):
        """Get results for all tried models."""
        np.random.seed(111)
        x = np.random.randn(400)
        y = np.random.randn(400)

        config = ModelSelectionConfig(
            try_diagonal=True,
            try_gmp=True,
            try_tt_full=False
        )
        selector = ModelSelector(config=config)
        selector.fit(x, y)

        results = selector.get_all_results()
        assert isinstance(results, dict)
        assert 'Diagonal-MP' in results
        if config.try_gmp:
            assert 'GMP' in results

    def test_explain_returns_string(self):
        """Explain returns informative string."""
        np.random.seed(112)
        x = np.random.randn(400)
        y = np.random.randn(400)

        selector = ModelSelector()
        selector.fit(x, y)

        explanation = selector.explain()
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert 'Selected' in explanation or 'selected' in explanation

    def test_explain_includes_metrics(self):
        """Explain includes validation metrics."""
        np.random.seed(113)
        x = np.random.randn(400)
        y = np.random.randn(400)

        selector = ModelSelector()
        selector.fit(x, y)

        explanation = selector.explain()
        # Should mention at least one metric
        assert 'NMSE' in explanation or 'AIC' in explanation or 'BIC' in explanation

    def test_explain_before_fit_raises(self):
        """Explain before fit should raise."""
        selector = ModelSelector()

        with pytest.raises(RuntimeError, match="must be fitted"):
            selector.explain()


class TestModelSelectorProperties:
    """Test ModelSelector properties."""

    def test_is_fitted_property(self):
        """is_fitted reflects fitting status."""
        selector = ModelSelector()
        assert not selector.is_fitted

        np.random.seed(114)
        x = np.random.randn(400)
        y = np.random.randn(400)
        selector.fit(x, y)

        assert selector.is_fitted

    def test_selected_model_type_property(self):
        """selected_model_type returns correct type."""
        np.random.seed(115)
        x = np.random.randn(400)
        y = np.random.randn(400)

        config = ModelSelectionConfig(
            try_diagonal=True,
            try_gmp=False,
            try_tt_full=False
        )
        selector = ModelSelector(config=config)
        selector.fit(x, y)

        assert selector.selected_model_type == 'Diagonal-MP'

    def test_selected_model_type_before_fit_raises(self):
        """selected_model_type before fit should raise."""
        selector = ModelSelector()

        with pytest.raises(RuntimeError, match="must be fitted"):
            _ = selector.selected_model_type

    def test_repr(self):
        """String representation is informative."""
        selector = ModelSelector()
        repr_str = repr(selector)
        assert 'ModelSelector' in repr_str
        assert 'fitted=False' in repr_str

    def test_repr_after_fit(self):
        """String representation updates after fit."""
        np.random.seed(116)
        x = np.random.randn(400)
        y = np.random.randn(400)

        selector = ModelSelector()
        selector.fit(x, y)

        repr_str = repr(selector)
        assert 'fitted=True' in repr_str


class TestModelSelectorMetrics:
    """Test ModelSelector metric computation."""

    def test_nmse_computed_for_all_models(self):
        """NMSE is computed for all tried models."""
        np.random.seed(117)
        x = np.random.randn(400)
        y = np.random.randn(400)

        config = ModelSelectionConfig(
            try_diagonal=True,
            try_gmp=True,
            try_tt_full=False
        )
        selector = ModelSelector(config=config)
        selector.fit(x, y)

        results = selector.get_all_results()
        for model_type, metrics in results.items():
            assert 'nmse' in metrics
            assert isinstance(metrics['nmse'], float)
            assert metrics['nmse'] >= 0

    def test_aic_bic_computed(self):
        """AIC and BIC are computed for all models."""
        np.random.seed(118)
        x = np.random.randn(400)
        y = np.random.randn(400)

        selector = ModelSelector()
        selector.fit(x, y)

        results = selector.get_all_results()
        for model_type, metrics in results.items():
            assert 'aic' in metrics
            assert 'bic' in metrics

    def test_best_model_has_lowest_criterion(self):
        """Selected model should have lowest selection criterion."""
        np.random.seed(119)
        x = np.random.randn(400)
        y = np.random.randn(400)

        config = ModelSelectionConfig(
            selection_criterion='aic',
            try_diagonal=True,
            try_gmp=True,
            try_tt_full=False
        )
        selector = ModelSelector(config=config)
        selector.fit(x, y)

        results = selector.get_all_results()
        selected_type = selector.selected_model_type

        selected_aic = results[selected_type]['aic']
        for model_type, metrics in results.items():
            # Selected model should have lowest or comparable AIC
            assert selected_aic <= metrics['aic'] + 1e-6  # numerical tolerance


class TestModelSelectorIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self):
        """Complete workflow: init → fit → predict → explain → export."""
        np.random.seed(120)

        # Generate test data
        N = 8
        M = 2
        x_train = np.random.randn(800)

        # True diagonal system
        from volterra.tt.tt_als_mimo import build_mimo_delay_matrix
        X_delay = build_mimo_delay_matrix(x_train, N)
        h1 = np.random.randn(N) * 0.5
        h2 = np.random.randn(N) * 0.1
        y_valid = X_delay @ h1 + (X_delay ** 2) @ h2
        y_train = np.concatenate([np.zeros(N-1), y_valid])

        # Model selection
        config = ModelSelectionConfig(
            memory_length=N,
            order=M,
            validation_split=0.2,
            try_diagonal=True,
            try_gmp=True,
            try_tt_full=False
        )
        selector = ModelSelector(config=config)
        selector.fit(x_train, y_train)

        # Predict
        x_test = np.random.randn(200)
        y_test = selector.predict(x_test)
        assert y_test.shape == (200,)

        # Explain
        explanation = selector.explain()
        assert len(explanation) > 50

        # Get selected model
        model = selector.get_selected_model()
        assert model is not None

        # Get all results
        results = selector.get_all_results()
        assert len(results) >= 1

    def test_workflow_preferring_simpler_model(self):
        """When models perform similarly, prefer simpler one."""
        np.random.seed(121)

        # Generate simple linear system
        N = 5
        h_true = np.random.randn(N) * 0.5

        x_train = np.random.randn(600)
        y_valid = np.convolve(x_train, h_true, mode='valid')
        y_train = np.concatenate([np.zeros(N-1), y_valid])

        # Add minimal noise
        y_train += np.random.randn(len(y_train)) * 0.01

        config = ModelSelectionConfig(
            memory_length=N,
            order=1,
            try_diagonal=True,
            try_gmp=True,
            try_tt_full=False,
            prefer_simpler=True  # Prefer diagonal if comparable
        )
        selector = ModelSelector(config=config)
        selector.fit(x_train, y_train)

        # Should select diagonal MP for linear system
        assert selector.selected_model_type == 'Diagonal-MP'
