"""
Tests for real TT-ALS implementation with diagonal Volterra.

These tests verify the production-ready TT-ALS solver:
1. Exact polynomial coefficient recovery
2. Memory polynomial identification
3. Convergence properties
4. Prediction accuracy with sliding windows
5. Numerical stability

Critical for validating Tasks 1 & 2 implementations.
"""

import numpy as np
from numpy.testing import assert_allclose

from volterra.models import TTVolterraConfig, TTVolterraIdentifier


class TestMemorylessPolynomialRecovery:
    """Test exact recovery of memoryless polynomial coefficients."""

    def test_linear_memoryless(self):
        """Recover y = a*x exactly."""
        np.random.seed(42)
        a = 2.5

        x = np.random.randn(500)
        y = a * x

        identifier = TTVolterraIdentifier(
            memory_length=1, order=1, ranks=[1, 1], config=TTVolterraConfig(max_iter=50, tol=1e-10)
        )
        identifier.fit(x, y)

        # Extract coefficient
        a_est = identifier.get_kernels().cores[0][0, 0, 0]

        assert_allclose(a_est, a, rtol=1e-6)
        assert identifier.fit_info_["per_output"][0]["converged"]

    def test_quadratic_memoryless(self):
        """Recover y = a1*x + a2*x^2 exactly."""
        np.random.seed(42)
        a1, a2 = 0.8, 0.2

        x = np.random.randn(500) * 0.5
        y = a1 * x + a2 * x**2

        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=2,
            ranks=[1, 1, 1],
            config=TTVolterraConfig(max_iter=50, tol=1e-10),
        )
        identifier.fit(x, y)

        # Extract coefficients
        a1_est = identifier.get_kernels().cores[0][0, 0, 0]
        a2_est = identifier.get_kernels().cores[1][0, 0, 0]

        assert_allclose([a1_est, a2_est], [a1, a2], rtol=1e-5)
        assert identifier.fit_info_["per_output"][0]["final_loss"] < 1e-15

    def test_cubic_memoryless(self):
        """Recover y = a1*x + a2*x^2 + a3*x^3 exactly."""
        np.random.seed(42)
        a1, a2, a3 = 0.7, 0.2, 0.05

        x = np.random.randn(500) * 0.5
        y = a1 * x + a2 * x**2 + a3 * x**3

        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=3,
            ranks=[1, 1, 1, 1],
            config=TTVolterraConfig(max_iter=100, tol=1e-10),
        )
        identifier.fit(x, y)

        coeffs_est = [identifier.get_kernels().cores[m][0, 0, 0] for m in range(3)]
        coeffs_true = [a1, a2, a3]

        assert_allclose(coeffs_est, coeffs_true, rtol=1e-4)

    def test_fifth_order_memoryless(self):
        """Test 5th-order polynomial recovery."""
        np.random.seed(42)
        coeffs_true = [0.8, 0.15, 0.05, 0.02, 0.01]

        x = np.random.randn(800) * 0.3
        y = sum(c * x ** (m + 1) for m, c in enumerate(coeffs_true))

        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=5,
            ranks=[1, 1, 1, 1, 1, 1],
            config=TTVolterraConfig(max_iter=150, tol=1e-10),
        )
        identifier.fit(x, y)

        coeffs_est = [identifier.get_kernels().cores[m][0, 0, 0] for m in range(5)]

        assert_allclose(coeffs_est, coeffs_true, rtol=1e-3)


class TestMemoryPolynomialRecovery:
    """Test recovery with memory (multi-tap filters)."""

    def test_linear_with_memory(self):
        """Recover FIR filter: y(t) = h0*x(t) + h1*x(t-1) + h2*x(t-2)."""
        np.random.seed(42)
        h_true = np.array([0.5, 0.3, 0.2])
        N = len(h_true)

        x = np.random.randn(500)
        y = np.zeros_like(x)

        # Generate output with memory
        for t in range(N - 1, len(x)):
            for i in range(N):
                y[t] += h_true[i] * x[t - i]

        identifier = TTVolterraIdentifier(
            memory_length=N,
            order=1,  # Linear only
            ranks=[1, 1],
            config=TTVolterraConfig(max_iter=50, tol=1e-10),
        )
        identifier.fit(x, y)

        h_est = identifier.get_kernels().cores[0][0, :, 0]

        assert_allclose(h_est, h_true, rtol=1e-2)

    def test_quadratic_with_memory(self):
        """Recover memory polynomial with quadratic terms."""
        np.random.seed(42)
        N = 3

        # Define true kernels
        h1 = np.array([0.6, 0.3, 0.1])
        h2 = np.array([0.1, 0.05, 0.02])

        x = np.random.randn(600) * 0.5
        y = np.zeros_like(x)

        # Generate output with memory
        for t in range(N - 1, len(x)):
            for i in range(N):
                y[t] += h1[i] * x[t - i] + h2[i] * x[t - i] ** 2

        identifier = TTVolterraIdentifier(
            memory_length=N,
            order=2,
            ranks=[1, 1, 1],
            config=TTVolterraConfig(max_iter=100, tol=1e-9),
        )
        identifier.fit(x, y)

        h1_est = identifier.get_kernels().cores[0][0, :, 0]
        h2_est = identifier.get_kernels().cores[1][0, :, 0]

        assert_allclose(h1_est, h1, rtol=1e-3)
        assert_allclose(h2_est, h2, rtol=1e-3)

    def test_long_memory(self):
        """Test with longer memory N=10."""
        np.random.seed(42)
        N = 10

        # Exponentially decaying kernel
        h1 = 0.5 ** np.arange(N)
        h1 /= np.sum(h1)  # Normalize

        x = np.random.randn(800)
        y = np.zeros_like(x)

        # Generate output with memory
        for t in range(N - 1, len(x)):
            for i in range(N):
                y[t] += h1[i] * x[t - i]

        identifier = TTVolterraIdentifier(
            memory_length=N, order=1, ranks=[1, 1], config=TTVolterraConfig(max_iter=100, tol=1e-8)
        )
        identifier.fit(x, y)

        h1_est = identifier.get_kernels().cores[0][0, :, 0]

        assert_allclose(h1_est, h1, rtol=1e-2)


class TestPredictionAccuracy:
    """Test prediction accuracy with sliding windows."""

    def test_predict_matches_fit_data(self):
        """Prediction on training data should have low error."""
        np.random.seed(42)
        x = np.random.randn(500) * 0.5
        y = 0.8 * x + 0.1 * x**2

        identifier = TTVolterraIdentifier(
            memory_length=1, order=2, ranks=[1, 1, 1], config=TTVolterraConfig(max_iter=50)
        )
        identifier.fit(x, y)

        y_pred = identifier.predict(x)

        # MSE should match final training loss
        mse = np.mean((y[0 : len(y_pred)] - y_pred) ** 2)
        final_loss = identifier.fit_info_["per_output"][0]["final_loss"]

        assert_allclose(mse, final_loss, rtol=1e-2)

    def test_predict_with_memory(self):
        """Prediction with memory should be accurate."""
        np.random.seed(42)
        N = 5
        h1 = np.array([0.5, 0.3, 0.15, 0.1, 0.05])

        x_train = np.random.randn(600)
        y_train = np.zeros_like(x_train)

        # Generate training output with memory
        for t in range(N - 1, len(x_train)):
            for i in range(N):
                y_train[t] += h1[i] * x_train[t - i]

        identifier = TTVolterraIdentifier(
            memory_length=N, order=1, ranks=[1, 1], config=TTVolterraConfig(max_iter=50)
        )
        identifier.fit(x_train, y_train)

        # Predict on new data
        x_test = np.random.randn(300)
        y_test_true = np.zeros_like(x_test)

        # Generate test output with memory
        for t in range(N - 1, len(x_test)):
            for i in range(N):
                y_test_true[t] += h1[i] * x_test[t - i]

        y_test_pred = identifier.predict(x_test)

        # Predict returns T - N + 1 samples, so compare valid portion
        y_test_valid = y_test_true[N - 1 :]
        mse = np.mean((y_test_valid - y_test_pred) ** 2)
        assert mse < 1e-10

    def test_predict_output_length(self):
        """Predict should return correct output length."""
        x_train = np.random.randn(500)
        y_train = x_train + 0.1 * x_train**2

        identifier = TTVolterraIdentifier(memory_length=10, order=2, ranks=[1, 1, 1])
        identifier.fit(x_train, y_train)

        x_test = np.random.randn(300)
        y_pred = identifier.predict(x_test)

        # Output length = T - N + 1
        assert len(y_pred) == 300 - 10 + 1

    def test_predict_memoryless_full_length(self):
        """Memoryless (N=1) prediction should match input length."""
        x = np.random.randn(500)
        y = 2 * x

        identifier = TTVolterraIdentifier(memory_length=1, order=1, ranks=[1, 1])
        identifier.fit(x, y)

        y_pred = identifier.predict(x)

        # For N=1, T_valid = T - 1 + 1 = T
        assert len(y_pred) == len(x)


class TestConvergenceProperties:
    """Test convergence behavior."""

    def test_convergence_indicator(self):
        """Convergence flag should be set correctly."""
        np.random.seed(42)
        x = np.random.randn(500)
        y = x + 0.1 * x**2

        # Should converge easily
        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=2,
            ranks=[1, 1, 1],
            config=TTVolterraConfig(max_iter=100, tol=1e-6),
        )
        identifier.fit(x, y)

        assert identifier.fit_info_["per_output"][0]["converged"]

    def test_max_iterations(self):
        """Should stop at max_iter if not converged."""
        np.random.seed(42)
        x = np.random.randn(200)
        y = x + 0.1 * x**2

        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=2,
            ranks=[1, 1, 1],
            config=TTVolterraConfig(max_iter=2, tol=1e-10),  # Very tight tol
        )
        identifier.fit(x, y)

        info = identifier.fit_info_["per_output"][0]
        assert info["iterations"] <= 2

    def test_loss_decreases(self):
        """Loss should decrease monotonically."""
        np.random.seed(42)
        x = np.random.randn(500)
        y = 0.8 * x + 0.2 * x**2

        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=2,
            ranks=[1, 1, 1],
            config=TTVolterraConfig(max_iter=50, verbose=False),
        )
        identifier.fit(x, y)

        loss_history = identifier.fit_info_["per_output"][0]["loss_history"]

        # Check loss decreases (or stays same)
        for i in range(1, len(loss_history)):
            assert loss_history[i] <= loss_history[i - 1] * 1.01  # Allow tiny numerical increase


class TestNumericalStability:
    """Test numerical stability and robustness."""

    def test_noisy_data(self):
        """Should work with moderate noise."""
        np.random.seed(42)
        x = np.random.randn(800)
        y_clean = 0.8 * x + 0.1 * x**2
        y_noisy = y_clean + 0.01 * np.random.randn(len(y_clean))  # 1% noise

        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=2,
            ranks=[1, 1, 1],
            config=TTVolterraConfig(max_iter=100, regularization=1e-6),
        )
        identifier.fit(x, y_noisy)

        # Should converge even with noise
        assert identifier.is_fitted
        assert identifier.fit_info_["per_output"][0]["final_loss"] < 1e-2

    def test_small_regularization(self):
        """Regularization parameter should be adjustable."""
        x = np.random.randn(500)
        y = x + 0.1 * x**2

        # Smaller regularization
        identifier = TTVolterraIdentifier(
            memory_length=1, order=2, ranks=[1, 1, 1], config=TTVolterraConfig(regularization=1e-10)
        )
        identifier.fit(x, y)

        assert identifier.is_fitted

    def test_high_order_stability(self):
        """High-order should remain stable."""
        np.random.seed(42)
        x = np.random.randn(1000) * 0.3  # Small amplitude for high powers
        y = sum(0.1 ** (m + 1) * x ** (m + 1) for m in range(5))

        identifier = TTVolterraIdentifier(
            memory_length=1,
            order=5,
            ranks=[1] * 6,
            config=TTVolterraConfig(max_iter=200, regularization=1e-7),
        )
        identifier.fit(x, y)

        assert identifier.is_fitted
        assert identifier.fit_info_["per_output"][0]["final_loss"] < 1e-4


class TestEdgeCasesReal:
    """Test edge cases with real solver."""

    def test_single_coefficient(self):
        """N=1, M=1 (single coefficient) should work."""
        x = np.random.randn(100)
        y = 2.0 * x

        identifier = TTVolterraIdentifier(memory_length=1, order=1, ranks=[1, 1])
        identifier.fit(x, y)

        coeff = identifier.get_kernels().cores[0][0, 0, 0]
        assert_allclose(coeff, 2.0, rtol=1e-5)

    def test_very_short_signal(self):
        """Short signals should work if T >= N."""
        x = np.random.randn(50)
        y = x + 0.1 * x**2

        identifier = TTVolterraIdentifier(memory_length=5, order=2, ranks=[1, 1, 1])
        identifier.fit(x, y)

        assert identifier.is_fitted

    def test_deterministic_signal(self):
        """Should work with deterministic (non-random) signals."""
        x = np.sin(2 * np.pi * np.arange(500) / 50)
        y = 0.8 * x + 0.2 * x**2

        identifier = TTVolterraIdentifier(memory_length=1, order=2, ranks=[1, 1, 1])
        identifier.fit(x, y)

        assert identifier.is_fitted
