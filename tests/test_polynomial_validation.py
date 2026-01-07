"""
Polynomial Validation Tests.

These tests verify that known polynomial systems can be perfectly modeled.
Error tolerance: <1% (if higher, implementation is WRONG).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from volterra import VolterraKernelFull, VolterraProcessorFull


class TestPolynomialValidation:
    """Validate kernel extraction and evaluation against known polynomial systems."""

    def test_memoryless_polynomial_evaluation(self):
        """
        Known memoryless polynomial: y = a₁x + a₂x² + a₃x³ + a₅x⁵

        Verify kernel evaluation produces exact polynomial output.
        """
        # Coefficients
        a1, a2, a3, a5 = 1.0, 0.1, 0.01, 0.005

        # Create kernel
        kernel = VolterraKernelFull.from_polynomial_coeffs(N=512, a1=a1, a2=a2, a3=a3, a5=a5)

        # Create processor
        processor = VolterraProcessorFull(kernel, sample_rate=48000, use_numba=False)

        # Test on range of inputs
        x_test = np.linspace(-0.5, 0.5, 1000)
        y_processor = np.zeros_like(x_test)

        processor.reset()
        for i, xi in enumerate(x_test):
            yi = processor.process_block(np.array([xi]))[0]
            y_processor[i] = yi

        # Expected output (direct polynomial evaluation)
        y_expected = a1 * x_test + a2 * x_test**2 + a3 * x_test**3 + a5 * x_test**5

        # Verify
        abs_error = np.max(np.abs(y_processor - y_expected))
        rel_error = abs_error / np.max(np.abs(y_expected))
        rel_error_percent = rel_error * 100

        assert (
            rel_error_percent < 1.0
        ), f"Polynomial evaluation error {rel_error_percent:.4f}% exceeds 1% tolerance"

        # Also check coefficient extraction
        assert_allclose(
            kernel.h1[0], a1, rtol=1e-10, atol=1e-12, err_msg="h1[0] coefficient mismatch"
        )
        assert_allclose(
            kernel.h2[0], a2, rtol=1e-10, atol=1e-12, err_msg="h2[0] coefficient mismatch"
        )
        assert_allclose(
            kernel.h3_diagonal[0], a3, rtol=1e-10, atol=1e-12, err_msg="h3[0] coefficient mismatch"
        )
        assert_allclose(
            kernel.h5_diagonal[0], a5, rtol=1e-10, atol=1e-12, err_msg="h5[0] coefficient mismatch"
        )

    def test_polynomial_all_orders(self):
        """Test polynomial with ALL orders 1-5."""
        coeffs = {"a1": 0.95, "a2": 0.12, "a3": 0.03, "a4": 0.008, "a5": 0.015}

        kernel = VolterraKernelFull.from_polynomial_coeffs(N=512, **coeffs)
        processor = VolterraProcessorFull(kernel, sample_rate=48000, use_numba=False)

        x_test = np.linspace(-0.4, 0.4, 500)
        y_processor = np.array([processor.process_block(np.array([xi]))[0] for xi in x_test])

        y_expected = (
            coeffs["a1"] * x_test
            + coeffs["a2"] * x_test**2
            + coeffs["a3"] * x_test**3
            + coeffs["a4"] * x_test**4
            + coeffs["a5"] * x_test**5
        )

        rel_error = np.max(np.abs(y_processor - y_expected)) / np.max(np.abs(y_expected))

        assert rel_error < 0.01, f"All-orders polynomial error {rel_error*100:.4f}% > 1%"

    def test_polynomial_on_audio_signal(self):
        """Test polynomial processing on realistic audio signal."""
        # Simple polynomial
        a1, a2, a3 = 0.9, 0.15, 0.05

        kernel = VolterraKernelFull.from_polynomial_coeffs(N=512, a1=a1, a2=a2, a3=a3)
        processor = VolterraProcessorFull(kernel, sample_rate=48000, use_numba=False)

        # Generate audio signal (1 second)
        t = np.linspace(0, 1.0, 48000)
        x_audio = 0.2 * (np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t))

        # Process
        y_processor = processor.process(x_audio, block_size=512)

        # Expected (sample-by-sample polynomial)
        y_expected = a1 * x_audio + a2 * x_audio**2 + a3 * x_audio**3

        # Verify
        max_error = np.max(np.abs(y_processor - y_expected))
        rms_error = np.sqrt(np.mean((y_processor - y_expected) ** 2))

        assert max_error < 1e-10, f"Audio polynomial max error {max_error:.2e} too large"
        assert rms_error < 1e-11, f"Audio polynomial RMS error {rms_error:.2e} too large"

    def test_edge_case_zero_input(self):
        """Zero input should produce zero output (linearity check)."""
        kernel = VolterraKernelFull.from_polynomial_coeffs(N=512, a1=1.0, a2=0.1, a3=0.01, a5=0.005)
        processor = VolterraProcessorFull(kernel, sample_rate=48000, use_numba=False)

        # Zero input
        x_zero = np.zeros(1000)
        y = processor.process(x_zero, block_size=256)

        assert np.max(np.abs(y)) < 1e-15, "Zero input should produce zero output"

    def test_edge_case_dc_input(self):
        """DC input should be processed correctly."""
        a1, a2, a3 = 1.0, 0.1, 0.01

        kernel = VolterraKernelFull.from_polynomial_coeffs(N=512, a1=a1, a2=a2, a3=a3)
        processor = VolterraProcessorFull(kernel, sample_rate=48000, use_numba=False)

        # DC input
        dc_value = 0.3
        x_dc = np.full(1000, dc_value)

        y = processor.process(x_dc, block_size=256)

        # Expected: polynomial evaluated at DC value
        expected = a1 * dc_value + a2 * dc_value**2 + a3 * dc_value**3

        # After transient (last 500 samples should be stable)
        y_steady = y[-500:]

        assert_allclose(
            y_steady, expected, rtol=1e-10, atol=1e-12, err_msg="DC input not processed correctly"
        )

    @pytest.mark.parametrize("amplitude", [0.1, 0.3, 0.5, 0.7])
    def test_polynomial_different_amplitudes(self, amplitude):
        """Polynomial should work correctly for different input amplitudes."""
        a1, a2, a3 = 1.0, 0.1, 0.01

        kernel = VolterraKernelFull.from_polynomial_coeffs(N=512, a1=a1, a2=a2, a3=a3)
        processor = VolterraProcessorFull(kernel, sample_rate=48000, use_numba=False)

        x_test = amplitude * np.array([0.0, 0.5, 1.0, -0.5, -1.0])
        y_processor = np.array([processor.process_block(np.array([xi]))[0] for xi in x_test])

        y_expected = a1 * x_test + a2 * x_test**2 + a3 * x_test**3

        assert_allclose(
            y_processor,
            y_expected,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Failed at amplitude {amplitude}",
        )
