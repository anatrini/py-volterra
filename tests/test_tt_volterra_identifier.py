"""
Tests for TT-Volterra identifier (high-level API).

These tests verify:
1. TTVolterraIdentifier API (fit, predict, get_kernels)
2. SISO and MIMO identification
3. Configuration and parameter validation
4. Error handling

Critical for STEP 4: TT-Volterra Identification
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from volterra.models import TTVolterraIdentifier, TTVolterraConfig


class TestTTVolterraIdentifierBasics:
    """Test basic TTVolterraIdentifier functionality."""

    def test_initialization(self):
        """Basic initialization should work."""
        identifier = TTVolterraIdentifier(
            memory_length=10,
            order=2,
            ranks=[1, 3, 1]
        )

        assert identifier.memory_length == 10
        assert identifier.order == 2
        assert identifier.ranks == [1, 3, 1]
        assert not identifier.is_fitted

    def test_initialization_with_config(self):
        """Initialization with custom config."""
        config = TTVolterraConfig(
            solver='mals',
            max_iter=50,
            tol=1e-5,
            verbose=False
        )
        identifier = TTVolterraIdentifier(
            memory_length=8,
            order=3,
            ranks=[1, 2, 2, 1],
            config=config
        )

        assert identifier.config.solver == 'mals'
        assert identifier.config.max_iter == 50

    def test_invalid_memory_length(self):
        """Memory length must be >= 1."""
        with pytest.raises(ValueError, match="memory_length"):
            TTVolterraIdentifier(
                memory_length=0,
                order=2,
                ranks=[1, 3, 1]
            )

    def test_invalid_order(self):
        """Order must be >= 1."""
        with pytest.raises(ValueError, match="order"):
            TTVolterraIdentifier(
                memory_length=10,
                order=0,
                ranks=[1]
            )

    def test_invalid_ranks_length(self):
        """Ranks length must match order+1."""
        with pytest.raises(ValueError, match="ranks"):
            TTVolterraIdentifier(
                memory_length=10,
                order=2,
                ranks=[1, 3]  # Should be 3 ranks
            )

    def test_invalid_boundary_ranks(self):
        """Boundary ranks must be 1."""
        with pytest.raises(ValueError, match="Boundary"):
            TTVolterraIdentifier(
                memory_length=10,
                order=2,
                ranks=[2, 3, 1]  # r_0 should be 1
            )

        with pytest.raises(ValueError, match="Boundary"):
            TTVolterraIdentifier(
                memory_length=10,
                order=2,
                ranks=[1, 3, 2]  # r_M should be 1
            )

    def test_repr(self):
        """String representation should show state."""
        identifier = TTVolterraIdentifier(
            memory_length=10,
            order=2,
            ranks=[1, 3, 1]
        )

        repr_str = repr(identifier)
        assert "N=10" in repr_str
        assert "M=2" in repr_str
        assert "not fitted" in repr_str


class TestTTVolterraFitting:
    """Test fitting functionality."""

    def test_fit_siso_basic(self):
        """Basic SISO fitting should work."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        identifier = TTVolterraIdentifier(
            memory_length=5,
            order=2,
            ranks=[1, 2, 1]
        )
        identifier.fit(x, y)

        assert identifier.is_fitted
        assert identifier.n_inputs_ == 1
        assert identifier.n_outputs_ == 1
        assert len(identifier.tt_models_) == 1

    def test_fit_mimo_basic(self):
        """Basic MIMO fitting should work."""
        np.random.seed(42)
        x = np.random.randn(100, 2)  # 2 inputs
        y = np.random.randn(100, 3)  # 3 outputs

        identifier = TTVolterraIdentifier(
            memory_length=5,
            order=2,
            ranks=[1, 2, 1]
        )
        identifier.fit(x, y)

        assert identifier.is_fitted
        assert identifier.n_inputs_ == 2
        assert identifier.n_outputs_ == 3
        assert len(identifier.tt_models_) == 3  # One per output

    def test_fit_returns_self(self):
        """fit() should return self for chaining."""
        x = np.random.randn(100)
        y = np.random.randn(100)

        identifier = TTVolterraIdentifier(
            memory_length=5,
            order=2,
            ranks=[1, 2, 1]
        )
        result = identifier.fit(x, y)

        assert result is identifier

    def test_fit_stores_info(self):
        """fit() should store fitting info."""
        x = np.random.randn(100)
        y = np.random.randn(100)

        identifier = TTVolterraIdentifier(
            memory_length=5,
            order=2,
            ranks=[1, 2, 1]
        )
        identifier.fit(x, y)

        assert identifier.fit_info_ is not None
        assert 'per_output' in identifier.fit_info_
        assert 'n_outputs' in identifier.fit_info_

    def test_fit_invalid_data_shape_mismatch(self):
        """Mismatched input/output lengths should raise error."""
        x = np.random.randn(100)
        y = np.random.randn(50)  # Wrong length

        identifier = TTVolterraIdentifier(
            memory_length=5,
            order=2,
            ranks=[1, 2, 1]
        )

        with pytest.raises(ValueError, match="same length"):
            identifier.fit(x, y)


class TestTTVolterraPrediction:
    """Test prediction functionality."""

    def test_predict_not_fitted(self):
        """predict() before fit() should raise error."""
        identifier = TTVolterraIdentifier(
            memory_length=5,
            order=2,
            ranks=[1, 2, 1]
        )
        x = np.random.randn(100)

        with pytest.raises(ValueError, match="not fitted"):
            identifier.predict(x)

    def test_predict_siso(self):
        """SISO prediction should return correct shape."""
        np.random.seed(42)
        x_train = np.random.randn(100)
        y_train = np.random.randn(100)

        identifier = TTVolterraIdentifier(
            memory_length=5,
            order=2,
            ranks=[1, 2, 1]
        )
        identifier.fit(x_train, y_train)

        x_test = np.random.randn(50)
        y_pred = identifier.predict(x_test)

        # Output should be 1D for SISO
        assert y_pred.ndim == 1
        # Length should account for memory
        assert len(y_pred) == 50 - 5 + 1

    def test_predict_mimo(self):
        """MIMO prediction should return correct shape."""
        np.random.seed(42)
        x_train = np.random.randn(100, 2)
        y_train = np.random.randn(100, 3)

        identifier = TTVolterraIdentifier(
            memory_length=5,
            order=2,
            ranks=[1, 2, 1]
        )
        identifier.fit(x_train, y_train)

        x_test = np.random.randn(50, 2)
        y_pred = identifier.predict(x_test)

        # Output should be 2D for MIMO
        assert y_pred.ndim == 2
        assert y_pred.shape == (50 - 5 + 1, 3)

    def test_predict_input_channel_mismatch(self):
        """Predict with wrong number of input channels should raise error."""
        x_train = np.random.randn(100, 2)
        y_train = np.random.randn(100)

        identifier = TTVolterraIdentifier(
            memory_length=5,
            order=2,
            ranks=[1, 2, 1]
        )
        identifier.fit(x_train, y_train)

        x_test_wrong = np.random.randn(50, 3)  # 3 inputs, expected 2

        with pytest.raises(ValueError, match="channels"):
            identifier.predict(x_test_wrong)

    def test_predict_insufficient_samples(self):
        """Predict with fewer samples than memory should raise error."""
        x_train = np.random.randn(100)
        y_train = np.random.randn(100)

        identifier = TTVolterraIdentifier(
            memory_length=10,
            order=2,
            ranks=[1, 2, 1]
        )
        identifier.fit(x_train, y_train)

        x_test_short = np.random.randn(5)  # Less than memory_length=10

        with pytest.raises(ValueError, match="memory_length"):
            identifier.predict(x_test_short)


class TestTTVolterraKernelExtraction:
    """Test kernel extraction."""

    def test_get_kernels_not_fitted(self):
        """get_kernels() before fit() should raise error."""
        identifier = TTVolterraIdentifier(
            memory_length=5,
            order=2,
            ranks=[1, 2, 1]
        )

        with pytest.raises(ValueError, match="not fitted"):
            identifier.get_kernels()

    def test_get_kernels_siso(self):
        """get_kernels() should return TT tensor for SISO."""
        x = np.random.randn(100)
        y = np.random.randn(100)

        identifier = TTVolterraIdentifier(
            memory_length=5,
            order=2,
            ranks=[1, 1, 1]  # Diagonal ranks
        )
        identifier.fit(x, y)

        tt_kernels = identifier.get_kernels(output_idx=0)

        assert tt_kernels.ndim == 2  # order=2
        assert tt_kernels.shape == (5, 5)  # memory_length=5
        assert tt_kernels.ranks == (1, 1, 1)  # Diagonal TT

    def test_get_kernels_mimo(self):
        """get_kernels() should return correct kernels for each output."""
        x = np.random.randn(100, 2)
        y = np.random.randn(100, 3)

        identifier = TTVolterraIdentifier(
            memory_length=5,
            order=2,
            ranks=[1, 1, 1]  # Diagonal ranks
        )
        identifier.fit(x, y)

        # Get kernels for each output
        for o in range(3):
            tt_kernels = identifier.get_kernels(output_idx=o)
            assert tt_kernels.ranks == (1, 1, 1)  # Diagonal TT

    def test_get_kernels_invalid_output_idx(self):
        """get_kernels() with invalid output index should raise error."""
        x = np.random.randn(100)
        y = np.random.randn(100)

        identifier = TTVolterraIdentifier(
            memory_length=5,
            order=2,
            ranks=[1, 1, 1]  # Diagonal ranks
        )
        identifier.fit(x, y)

        with pytest.raises(ValueError, match="output_idx"):
            identifier.get_kernels(output_idx=5)  # Only 1 output


class TestTTVolterraConfig:
    """Test TTVolterraConfig."""

    def test_config_defaults(self):
        """Default config should have sensible values."""
        config = TTVolterraConfig()

        assert config.solver == 'als'
        assert config.max_iter == 100
        assert config.tol == 1e-6
        assert not config.verbose
        assert not config.diagonal_only

    def test_config_invalid_solver(self):
        """Invalid solver should raise error."""
        with pytest.raises(ValueError, match="Solver"):
            TTVolterraConfig(solver='invalid')

    def test_config_rank_adaptation_auto_mals(self):
        """rank_adaptation=True should set solver='mals'."""
        config = TTVolterraConfig(
            solver='als',
            rank_adaptation=True
        )

        # Should auto-switch to mals
        assert config.solver == 'mals'

    def test_config_custom_parameters(self):
        """Custom config parameters should be set."""
        config = TTVolterraConfig(
            solver='mals',
            max_iter=50,
            tol=1e-5,
            regularization=1e-6,
            max_rank=8,
            rank_tol=1e-3,
            verbose=True,
            diagonal_only=True
        )

        assert config.solver == 'mals'
        assert config.max_iter == 50
        assert config.tol == 1e-5
        assert config.regularization == 1e-6
        assert config.max_rank == 8
        assert config.rank_tol == 1e-3
        assert config.verbose
        assert config.diagonal_only
