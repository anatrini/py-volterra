"""
Tests for TTVolterraMIMO (full TT-Volterra with arbitrary ranks).

Tests the public API for general MIMO Volterra system identification
using Tensor-Train decomposition with arbitrary ranks.
"""

import numpy as np
import pytest

from volterra.models import TTVolterraFullConfig, TTVolterraMIMO


class TestTTVolterraMIMOInitialization:
    """Test TTVolterraMIMO initialization and parameter validation."""

    def test_initialization_valid(self):
        """Valid initialization should succeed."""
        model = TTVolterraMIMO(memory_length=10, order=3, ranks=[1, 2, 2, 1])
        assert model.memory_length == 10
        assert model.order == 3
        assert model.ranks == [1, 2, 2, 1]
        assert not model.is_fitted
        assert not model.is_diagonal

    def test_initialization_diagonal(self):
        """Diagonal ranks should be detected."""
        model = TTVolterraMIMO(memory_length=5, order=2, ranks=[1, 1, 1])
        assert model.is_diagonal

    def test_initialization_with_config(self):
        """Initialization with custom config."""
        config = TTVolterraFullConfig(max_iter=50, tol=1e-8, verbose=True)
        model = TTVolterraMIMO(memory_length=10, order=2, ranks=[1, 3, 1], config=config)
        assert model.config.max_iter == 50
        assert model.config.tol == 1e-8
        assert model.config.verbose is True

    def test_initialization_invalid_memory(self):
        """Should reject invalid memory length."""
        with pytest.raises(ValueError, match="memory_length must be >= 1"):
            TTVolterraMIMO(memory_length=0, order=2, ranks=[1, 1, 1])

    def test_initialization_invalid_order(self):
        """Should reject invalid order."""
        with pytest.raises(ValueError, match="order must be >= 1"):
            TTVolterraMIMO(memory_length=5, order=0, ranks=[1, 1])

    def test_initialization_wrong_number_of_ranks(self):
        """Should reject wrong number of ranks."""
        with pytest.raises(ValueError, match="Need 3 ranks for order 2"):
            TTVolterraMIMO(memory_length=5, order=2, ranks=[1, 1])

    def test_initialization_invalid_boundary_ranks(self):
        """Should reject non-1 boundary ranks."""
        with pytest.raises(ValueError, match="Boundary ranks must be 1"):
            TTVolterraMIMO(memory_length=5, order=2, ranks=[2, 3, 1])

        with pytest.raises(ValueError, match="Boundary ranks must be 1"):
            TTVolterraMIMO(memory_length=5, order=2, ranks=[1, 3, 2])


class TestTTVolterraMIMOFitting:
    """Test TTVolterraMIMO fitting."""

    def test_fit_siso_linear(self):
        """Fit simple SISO linear system."""
        np.random.seed(42)

        # Generate linear system
        N = 5
        M = 1
        h_true = np.random.randn(N)

        x = np.random.randn(500)
        from volterra.tt.tt_als_mimo import build_mimo_delay_matrix

        X_delay = build_mimo_delay_matrix(x, N)
        y_valid = X_delay @ h_true

        # Pad y to same length as x (first N-1 samples are don't-care)
        y = np.concatenate([np.zeros(N - 1), y_valid])

        # Fit
        model = TTVolterraMIMO(
            memory_length=N,
            order=M,
            ranks=[1, 1],
            config=TTVolterraFullConfig(max_iter=50, tol=1e-8, verbose=False),
        )
        model.fit(x, y)

        assert model.is_fitted
        assert model.n_inputs_ == 1
        assert model.n_outputs_ == 1

        # Check fit quality
        diag = model.diagnostics()
        assert diag["final_loss"] < 1e-8

    def test_fit_siso_quadratic(self):
        """Fit SISO quadratic system with ranks > 1."""
        np.random.seed(42)

        N = 5
        M = 2

        x = np.random.randn(500) * 0.5
        y_valid = x[N - 1 :] + 0.1 * x[N - 1 :] ** 2
        # Pad y to match x length
        y = np.concatenate([np.zeros(N - 1), y_valid])

        # Fit with rank-2
        model = TTVolterraMIMO(
            memory_length=N,
            order=M,
            ranks=[1, 2, 1],
            config=TTVolterraFullConfig(max_iter=100, tol=1e-6, verbose=False),
        )
        model.fit(x, y)

        assert model.is_fitted
        assert model.n_inputs_ == 1
        assert model.n_outputs_ == 1

        # Should converge or achieve reasonable fit
        diag = model.diagnostics()
        assert diag["iterations"] > 0
        assert diag["final_loss"] < 1.0

    def test_fit_mimo_two_inputs(self):
        """Fit MIMO system with 2 inputs."""
        np.random.seed(42)

        N = 5
        M = 1
        I = 2

        # Linear combination of two inputs
        h1 = np.random.randn(N) * 0.8
        h2 = np.random.randn(N) * 0.6

        x = np.random.randn(500, I)
        from volterra.tt.tt_als_mimo import build_mimo_delay_matrix

        X_delay = build_mimo_delay_matrix(x, N)

        y_valid = X_delay[:, :N] @ h1 + X_delay[:, N:] @ h2
        # Pad y to match x length
        y = np.concatenate([np.zeros(N - 1), y_valid])

        # Fit
        model = TTVolterraMIMO(
            memory_length=N,
            order=M,
            ranks=[1, 1],
            config=TTVolterraFullConfig(max_iter=100, tol=1e-8, verbose=False),
        )
        model.fit(x, y)

        assert model.is_fitted
        assert model.n_inputs_ == 2
        assert model.n_outputs_ == 1

        # Check fit quality
        diag = model.diagnostics()
        assert diag["final_loss"] < 1e-6

    def test_fit_multi_output(self):
        """Fit system with multiple outputs."""
        np.random.seed(42)

        N = 5
        M = 1

        x = np.random.randn(200)
        y = np.column_stack(
            [np.random.randn(200) * 0.5, np.random.randn(200) * 0.3]  # Output 1  # Output 2
        )

        model = TTVolterraMIMO(
            memory_length=N,
            order=M,
            ranks=[1, 1],
            config=TTVolterraFullConfig(max_iter=50, verbose=False),
        )
        model.fit(x, y)

        assert model.is_fitted
        assert model.n_inputs_ == 1
        assert model.n_outputs_ == 2

        # Should have 2 sets of cores
        assert len(model.cores_) == 2

    def test_fit_sets_fitted_flag(self):
        """Fit should set is_fitted flag."""
        np.random.seed(42)

        model = TTVolterraMIMO(memory_length=5, order=1, ranks=[1, 1])
        assert not model.is_fitted

        x = np.random.randn(100)
        y = np.random.randn(100)
        model.fit(x, y)

        assert model.is_fitted


class TestTTVolterraMIMOPrediction:
    """Test TTVolterraMIMO prediction."""

    def test_predict_before_fit_raises(self):
        """Predict before fit should raise error."""
        model = TTVolterraMIMO(memory_length=5, order=1, ranks=[1, 1])
        x = np.random.randn(100)

        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(x)

    def test_predict_siso_shape(self):
        """Predict should return correct shape for SISO."""
        np.random.seed(42)

        N = 5
        x = np.random.randn(200)
        y = np.random.randn(200)

        model = TTVolterraMIMO(memory_length=N, order=1, ranks=[1, 1])
        model.fit(x[:100], y[:100])

        y_pred = model.predict(x[:50])

        # Should return 1D array for SISO
        assert y_pred.ndim == 1
        # T_valid = 50 - 5 + 1 = 46
        assert len(y_pred) == 46

    def test_predict_mimo_shape(self):
        """Predict should return correct shape for MIMO."""
        np.random.seed(42)

        N = 5
        x = np.random.randn(200, 2)  # 2 inputs
        y = np.random.randn(200)

        model = TTVolterraMIMO(memory_length=N, order=1, ranks=[1, 1])
        model.fit(x[:100], y[:100])

        y_pred = model.predict(x[:50])

        assert y_pred.ndim == 1
        assert len(y_pred) == 46

    def test_predict_multi_output_shape(self):
        """Predict should return correct shape for multi-output."""
        np.random.seed(42)

        N = 5
        x = np.random.randn(200)
        y = np.random.randn(200, 2)  # 2 outputs

        model = TTVolterraMIMO(memory_length=N, order=1, ranks=[1, 1])
        model.fit(x[:100], y[:100])

        y_pred = model.predict(x[:50])

        # Should return 2D for multi-output
        assert y_pred.ndim == 2
        assert y_pred.shape == (46, 2)

    def test_predict_wrong_input_channels(self):
        """Predict with wrong number of input channels should raise."""
        np.random.seed(42)

        x_train = np.random.randn(100, 2)  # 2 inputs
        y_train = np.random.randn(100)

        model = TTVolterraMIMO(memory_length=5, order=1, ranks=[1, 1])
        model.fit(x_train, y_train)

        x_test = np.random.randn(50)  # 1 input (wrong!)

        with pytest.raises(ValueError, match="Input has 1 channels, but model was fitted with 2"):
            model.predict(x_test)

    def test_predict_insufficient_samples(self):
        """Predict with insufficient samples should raise."""
        np.random.seed(42)

        N = 10
        x_train = np.random.randn(100)
        y_train = np.random.randn(100)

        model = TTVolterraMIMO(memory_length=N, order=1, ranks=[1, 1])
        model.fit(x_train, y_train)

        x_test = np.random.randn(5)  # Too short!

        with pytest.raises(ValueError, match="Input has 5 samples, but memory_length=10"):
            model.predict(x_test)


class TestTTVolterraMIMOAccessors:
    """Test accessor methods (get_cores, diagnostics, export_model)."""

    def test_get_cores(self):
        """get_cores should return TT cores."""
        np.random.seed(42)

        model = TTVolterraMIMO(memory_length=5, order=2, ranks=[1, 2, 1])
        x = np.random.randn(100)
        y = np.random.randn(100)
        model.fit(x, y)

        cores = model.get_cores(output_idx=0)

        assert isinstance(cores, list)
        assert len(cores) == 2  # order = 2
        # Check shapes: (1, mode, 2), (2, mode, 1)
        assert cores[0].shape[0] == 1
        assert cores[0].shape[2] == 2
        assert cores[1].shape[0] == 2
        assert cores[1].shape[2] == 1

    def test_get_cores_before_fit_raises(self):
        """get_cores before fit should raise."""
        model = TTVolterraMIMO(memory_length=5, order=1, ranks=[1, 1])

        with pytest.raises(ValueError, match="Model not fitted"):
            model.get_cores()

    def test_get_cores_invalid_output_idx(self):
        """get_cores with invalid output index should raise."""
        np.random.seed(42)

        model = TTVolterraMIMO(memory_length=5, order=1, ranks=[1, 1])
        x = np.random.randn(100)
        y = np.random.randn(100)
        model.fit(x, y)

        with pytest.raises(ValueError, match="output_idx must be in"):
            model.get_cores(output_idx=5)

    def test_diagnostics(self):
        """diagnostics should return training info."""
        np.random.seed(42)

        model = TTVolterraMIMO(
            memory_length=5,
            order=2,
            ranks=[1, 2, 1],
            config=TTVolterraFullConfig(max_iter=20, verbose=False),
        )
        x = np.random.randn(100)
        y = np.random.randn(100)
        model.fit(x, y)

        diag = model.diagnostics()

        assert "loss_history" in diag
        assert "converged" in diag
        assert "final_loss" in diag
        assert "iterations" in diag
        assert "ranks" in diag
        assert "total_parameters" in diag

        assert isinstance(diag["loss_history"], list)
        assert len(diag["loss_history"]) > 0
        assert diag["iterations"] <= 20
        assert diag["ranks"] == [1, 2, 1]

    def test_diagnostics_before_fit_raises(self):
        """diagnostics before fit should raise."""
        model = TTVolterraMIMO(memory_length=5, order=1, ranks=[1, 1])

        with pytest.raises(ValueError, match="Model not fitted"):
            model.diagnostics()

    def test_export_model(self):
        """export_model should return all necessary data."""
        np.random.seed(42)

        model = TTVolterraMIMO(memory_length=5, order=2, ranks=[1, 2, 1])
        x = np.random.randn(100)
        y = np.random.randn(100)
        model.fit(x, y)

        export = model.export_model()

        assert "cores" in export
        assert "ranks" in export
        assert "memory_length" in export
        assert "order" in export
        assert "n_inputs" in export
        assert "metadata" in export

        assert export["memory_length"] == 5
        assert export["order"] == 2
        assert export["ranks"] == [1, 2, 1]
        assert export["n_inputs"] == 1
        assert len(export["cores"]) == 2

    def test_export_model_before_fit_raises(self):
        """export_model before fit should raise."""
        model = TTVolterraMIMO(memory_length=5, order=1, ranks=[1, 1])

        with pytest.raises(ValueError, match="Model not fitted"):
            model.export_model()


class TestTTVolterraMIMOProperties:
    """Test model properties."""

    def test_is_diagonal_true(self):
        """is_diagonal should return True for all ranks = 1."""
        model = TTVolterraMIMO(memory_length=5, order=3, ranks=[1, 1, 1, 1])
        assert model.is_diagonal

    def test_is_diagonal_false(self):
        """is_diagonal should return False for ranks > 1."""
        model = TTVolterraMIMO(memory_length=5, order=3, ranks=[1, 2, 2, 1])
        assert not model.is_diagonal

    def test_total_parameters_before_fit(self):
        """total_parameters should estimate parameters before fit."""
        model = TTVolterraMIMO(memory_length=10, order=2, ranks=[1, 3, 1])

        # Parameters: (1*10*3) + (3*10*1) = 30 + 30 = 60
        assert model.total_parameters == 60

    def test_total_parameters_after_fit(self):
        """total_parameters should count actual parameters after fit."""
        np.random.seed(42)

        model = TTVolterraMIMO(memory_length=5, order=2, ranks=[1, 2, 1])
        x = np.random.randn(100)
        y = np.random.randn(100)
        model.fit(x, y)

        # Core 0: (1, 5, 2) = 10 params
        # Core 1: (2, 5, 1) = 10 params
        # Total: 20 params
        assert model.total_parameters == 20

    def test_total_parameters_mimo_before_fit(self):
        """total_parameters for MIMO before fit."""
        model = TTVolterraMIMO(memory_length=5, order=2, ranks=[1, 2, 1])
        model.n_inputs_ = 2  # Simulate MIMO

        # mode_size = 2 * 5 = 10
        # Core 0: (1, 10, 2) = 20
        # Core 1: (2, 10, 1) = 20
        # Total: 40
        assert model.total_parameters == 40

    def test_repr_not_fitted(self):
        """__repr__ before fit."""
        model = TTVolterraMIMO(memory_length=10, order=2, ranks=[1, 3, 1])
        repr_str = repr(model)

        assert "TTVolterraMIMO" in repr_str
        assert "N=10" in repr_str
        assert "M=2" in repr_str
        assert "ranks=[1, 3, 1]" in repr_str
        assert "not fitted" in repr_str

    def test_repr_fitted(self):
        """__repr__ after fit."""
        np.random.seed(42)

        model = TTVolterraMIMO(memory_length=5, order=1, ranks=[1, 1])
        x = np.random.randn(100)
        y = np.random.randn(100)
        model.fit(x, y)

        repr_str = repr(model)
        assert "fitted" in repr_str
        assert "diagonal" in repr_str  # All ranks = 1


class TestTTVolterraMIMOIntegration:
    """Integration tests with realistic scenarios."""

    def test_fit_predict_roundtrip_siso(self):
        """Full fit-predict roundtrip for SISO."""
        np.random.seed(42)

        N = 5
        M = 2

        # Generate data
        x = np.random.randn(500) * 0.5
        from volterra.tt.tt_als_mimo import build_mimo_delay_matrix

        X_delay = build_mimo_delay_matrix(x, N)
        h1 = np.random.randn(N) * 0.8
        h2 = np.random.randn(N) * 0.15
        y_valid = X_delay @ h1 + (X_delay**2) @ h2

        # Pad y to match x length
        y = np.concatenate([np.zeros(N - 1), y_valid])

        # Split train/test
        x_train, x_test = x[:400], x[400:]
        y_train = y[:400]

        # Fit
        model = TTVolterraMIMO(
            memory_length=N,
            order=M,
            ranks=[1, 1, 1],
            config=TTVolterraFullConfig(max_iter=100, tol=1e-8, verbose=False),
        )
        model.fit(x_train, y_train)

        # Predict
        y_pred = model.predict(x_test)

        # Should have correct length
        assert len(y_pred) == len(x_test) - N + 1

    def test_fit_predict_roundtrip_mimo(self):
        """Full fit-predict roundtrip for MIMO."""
        np.random.seed(42)

        N = 5
        M = 1
        I = 2

        # Generate MIMO data
        x = np.random.randn(500, I) * 0.5
        from volterra.tt.tt_als_mimo import build_mimo_delay_matrix

        X_delay = build_mimo_delay_matrix(x, N)

        h1 = np.random.randn(N) * 0.7
        h2 = np.random.randn(N) * 0.5
        y_valid = X_delay[:, :N] @ h1 + X_delay[:, N:] @ h2

        # Pad y to match x length
        y = np.concatenate([np.zeros(N - 1), y_valid])

        # Split
        x_train, x_test = x[:400], x[400:]
        y_train = y[:400]

        # Fit
        model = TTVolterraMIMO(
            memory_length=N,
            order=M,
            ranks=[1, 1],
            config=TTVolterraFullConfig(max_iter=100, tol=1e-8, verbose=False),
        )
        model.fit(x_train, y_train)

        # Predict
        y_pred = model.predict(x_test)

        assert len(y_pred) == len(x_test) - N + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
