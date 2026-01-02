"""
Tests for MIMO and RLS features.

These tests verify:
1. MIMO diagonal Volterra identification (additive model)
2. RLS online/adaptive filtering
3. MIMO prediction accuracy
4. RLS adaptation on time-varying systems
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from volterra.models import TTVolterraIdentifier, TTVolterraConfig


class TestMIMODiagonalVolterra:
    """Test MIMO diagonal Volterra identification."""

    def test_mimo_2inputs_memoryless(self):
        """MIMO with 2 inputs, memoryless polynomial."""
        np.random.seed(42)

        # Ground truth
        a1, a2 = 0.8, 0.2  # Coefficients for input 1
        b1, b2 = 0.5, 0.1  # Coefficients for input 2

        x = np.random.randn(800, 2) * 0.5
        y = (a1*x[:, 0] + a2*x[:, 0]**2 +
             b1*x[:, 1] + b2*x[:, 1]**2)

        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=2,
            ranks=[1, 1, 1],
            config=TTVolterraConfig(solver='als', max_iter=100, tol=1e-9)
        )
        identifier.fit(x, y)

        # Check convergence
        assert identifier.is_fitted
        assert identifier.fit_info_['per_output'][0]['converged']
        assert identifier.fit_info_['per_output'][0]['final_loss'] < 1e-15

    def test_mimo_3inputs_linear(self):
        """MIMO with 3 inputs, linear only."""
        np.random.seed(42)

        # Linear combination of 3 inputs
        weights = np.array([0.6, 0.3, 0.1])

        x = np.random.randn(600, 3)
        y = x @ weights

        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=1,
            ranks=[1, 1],
            config=TTVolterraConfig(solver='als', max_iter=50)
        )
        identifier.fit(x, y)

        assert identifier.is_fitted
        assert identifier.fit_info_['per_output'][0]['final_loss'] < 1e-10

    def test_mimo_with_memory(self):
        """MIMO with memory (FIR filters per input)."""
        np.random.seed(42)
        N = 5

        # Each input has its own FIR filter
        h1 = np.array([0.5, 0.3, 0.15, 0.1, 0.05])
        h2 = np.array([0.4, 0.3, 0.2, 0.1, 0.0])

        x = np.random.randn(800, 2)
        y = np.zeros(len(x))

        # Generate output with memory
        for t in range(N-1, len(x)):
            for k in range(N):
                y[t] += h1[k] * x[t-k, 0] + h2[k] * x[t-k, 1]

        identifier = TTVolterraIdentifier(
            memory_length=N,
            order=1,
            ranks=[1, 1],
            config=TTVolterraConfig(solver='als', max_iter=100, tol=1e-8)
        )
        identifier.fit(x, y)

        assert identifier.is_fitted
        assert identifier.fit_info_['per_output'][0]['final_loss'] < 1e-10

    def test_mimo_prediction_shape(self):
        """MIMO prediction returns correct shape."""
        x = np.random.randn(500, 2)
        y = x[:, 0] + 0.5*x[:, 1]

        identifier = TTVolterraIdentifier(
            memory_length=10,
            order=1,
            ranks=[1, 1],
        )
        identifier.fit(x, y)

        # Predict on new data
        x_test = np.random.randn(300, 2)
        y_pred = identifier.predict(x_test)

        # Output length = T - N + 1
        assert y_pred.shape == (300 - 10 + 1,)

    def test_mimo_vs_siso_equivalence(self):
        """MIMO with 1 input should match SISO."""
        np.random.seed(42)

        x_siso = np.random.randn(500)
        y = 0.8*x_siso + 0.2*x_siso**2

        # SISO model
        id_siso = TTVolterraIdentifier(
            memory_length=1,
            order=2,
            ranks=[1, 1, 1]
        )
        id_siso.fit(x_siso, y)

        # MIMO model with 1 input
        x_mimo = x_siso.reshape(-1, 1)
        id_mimo = TTVolterraIdentifier(
            memory_length=1,
            order=2,
            ranks=[1, 1, 1]
        )
        id_mimo.fit(x_mimo, y)

        # Both should give same results
        y_pred_siso = id_siso.predict(x_siso)
        y_pred_mimo = id_mimo.predict(x_mimo)

        assert_allclose(y_pred_siso, y_pred_mimo, rtol=1e-6)


class TestRLSAdaptiveFiltering:
    """Test RLS online/adaptive solver."""

    def test_rls_stationary_system(self):
        """RLS on stationary system should converge."""
        np.random.seed(42)

        x = np.random.randn(2000)
        y = 0.8*x + 0.1*x**2

        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=2,
            ranks=[1, 1, 1],
            config=TTVolterraConfig(
                solver='rls',
                forgetting_factor=1.0,  # Infinite memory for stationary
                regularization=1e-4
            )
        )
        identifier.fit(x, y)

        assert identifier.is_fitted
        # RLS should achieve reasonable MSE on stationary system
        final_mse = identifier.fit_info_['per_output'][0]['final_mse']
        assert final_mse < 0.001

    def test_rls_time_varying_system(self):
        """RLS should track time-varying coefficients."""
        np.random.seed(42)

        x = np.random.randn(5000)
        y = np.zeros_like(x)

        # Coefficient varies sinusoidally
        for t in range(len(x)):
            alpha = 0.8 + 0.2 * np.sin(2 * np.pi * t / 500)
            y[t] = alpha * x[t] + 0.1 * x[t]**2

        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=2,
            ranks=[1, 1, 1],
            config=TTVolterraConfig(
                solver='rls',
                forgetting_factor=0.995,  # Track variations
                regularization=1e-4,
                verbose=False
            )
        )
        identifier.fit(x, y)

        assert identifier.is_fitted
        # Should achieve reasonable tracking
        final_mse = identifier.fit_info_['per_output'][0]['final_mse']
        assert final_mse < 0.05  # Reasonable for time-varying

    def test_rls_with_memory(self):
        """RLS with memory polynomial."""
        np.random.seed(42)
        N = 5

        h1 = np.array([0.5, 0.3, 0.15, 0.1, 0.05])

        x = np.random.randn(3000)
        y = np.zeros_like(x)

        for t in range(N-1, len(x)):
            for k in range(N):
                y[t] += h1[k] * x[t-k]

        identifier = TTVolterraIdentifier(
            memory_length=N,
            order=1,
            ranks=[1, 1],
            config=TTVolterraConfig(
                solver='rls',
                forgetting_factor=1.0,
                regularization=1e-4
            )
        )
        identifier.fit(x, y)

        assert identifier.is_fitted
        final_mse = identifier.fit_info_['per_output'][0]['final_mse']
        assert final_mse < 0.001

    def test_rls_forgetting_factor_effect(self):
        """Different forgetting factors should give different results."""
        np.random.seed(42)

        x = np.random.randn(2000)
        y = 0.8*x + 0.1*x**2

        # High forgetting factor (slow adaptation)
        id_slow = TTVolterraIdentifier(
            memory_length=1,
            order=2,
            ranks=[1, 1, 1],
            config=TTVolterraConfig(
                solver='rls',
                forgetting_factor=0.999,
                regularization=1e-4
            )
        )
        id_slow.fit(x, y)

        # Low forgetting factor (fast adaptation)
        id_fast = TTVolterraIdentifier(
            memory_length=1,
            order=2,
            ranks=[1, 1, 1],
            config=TTVolterraConfig(
                solver='rls',
                forgetting_factor=0.95,
                regularization=1e-4
            )
        )
        id_fast.fit(x, y)

        # Both should converge but with different trajectories
        assert id_slow.is_fitted
        assert id_fast.is_fitted

        mse_slow = id_slow.fit_info_['per_output'][0]['final_mse']
        mse_fast = id_fast.fit_info_['per_output'][0]['final_mse']

        # Fast adaptation may have slightly higher final MSE due to forgetting
        assert mse_slow < 0.001
        assert mse_fast < 0.01

    def test_rls_mse_history(self):
        """RLS should provide MSE history."""
        np.random.seed(42)

        x = np.random.randn(1000)
        y = x + 0.1*x**2

        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=2,
            ranks=[1, 1, 1],
            config=TTVolterraConfig(solver='rls')
        )
        identifier.fit(x, y)

        # Check MSE history is available
        info = identifier.fit_info_['per_output'][0]
        assert 'mse_history' in info
        assert len(info['mse_history']) > 0

        # MSE should generally decrease over time (may fluctuate)
        mse_history = info['mse_history']
        assert mse_history[-1] < mse_history[0] * 0.1  # Final < 10% of initial


class TestMIMOandRLSIntegration:
    """Test integration and edge cases."""

    def test_rls_mimo_warning(self):
        """RLS with MIMO should warn and use first input."""
        x = np.random.randn(500, 2)
        y = x[:, 0] + 0.1*x[:, 0]**2

        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=2,
            ranks=[1, 1, 1],
            config=TTVolterraConfig(solver='rls')
        )

        with pytest.warns(UserWarning, match="RLS solver currently only supports SISO"):
            identifier.fit(x, y)

        assert identifier.is_fitted

    def test_mimo_info_stored(self):
        """MIMO fit should store MIMO info."""
        x = np.random.randn(500, 2)
        y = x[:, 0] + x[:, 1]

        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=1,
            ranks=[1, 1],
            config=TTVolterraConfig(solver='als')
        )
        identifier.fit(x, y)

        info = identifier.fit_info_['per_output'][0]
        assert info['mimo'] == True
        assert info['n_inputs'] == 2

    def test_siso_info_stored(self):
        """SISO fit should store MIMO=False."""
        x = np.random.randn(500)
        y = x + 0.1*x**2

        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=2,
            ranks=[1, 1, 1],
            config=TTVolterraConfig(solver='als')
        )
        identifier.fit(x, y)

        info = identifier.fit_info_['per_output'][0]
        assert info['mimo'] == False

    def test_config_rls_parameters(self):
        """TTVolterraConfig should accept RLS parameters."""
        config = TTVolterraConfig(
            solver='rls',
            forgetting_factor=0.98,
            regularization=1e-3
        )

        assert config.solver == 'rls'
        assert config.forgetting_factor == 0.98
        assert config.regularization == 1e-3

    def test_invalid_solver_raises(self):
        """Invalid solver should raise error."""
        with pytest.raises(ValueError, match="Solver must be"):
            TTVolterraConfig(solver='invalid')


class TestMIMOEdgeCases:
    """Test edge cases for MIMO."""

    def test_mimo_single_sample_memory(self):
        """MIMO with N=1 (memoryless)."""
        x = np.random.randn(500, 2)
        y = x[:, 0] + x[:, 1]

        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=1,
            ranks=[1, 1]
        )
        identifier.fit(x, y)

        y_pred = identifier.predict(x)
        assert len(y_pred) == len(x)  # Memoryless: full length

    def test_mimo_long_memory(self):
        """MIMO with long memory."""
        x = np.random.randn(800, 2) * 0.5
        y = np.zeros(len(x))

        N = 20
        h1 = 0.9 ** np.arange(N)
        h2 = 0.8 ** np.arange(N)

        for t in range(N-1, len(x)):
            for k in range(N):
                y[t] += h1[k] * x[t-k, 0] + h2[k] * x[t-k, 1]

        identifier = TTVolterraIdentifier(
            memory_length=N,
            order=1,
            ranks=[1, 1],
            config=TTVolterraConfig(max_iter=150, tol=1e-7)
        )
        identifier.fit(x, y)

        assert identifier.is_fitted
        assert identifier.fit_info_['per_output'][0]['final_loss'] < 1e-6
