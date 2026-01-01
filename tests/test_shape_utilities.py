"""
Tests for MIMO shape validation and canonicalization utilities.

These tests verify:
1. Input/output canonicalization to (T, I) / (T, O) format
2. Shape validation for MIMO data
3. Dimension inference (T, I, O)
4. Error handling for invalid shapes

Critical for STEP 2: MIMO Data Conventions
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from volterra.utils.shapes import (
    canonicalize_input,
    canonicalize_output,
    validate_mimo_data,
    infer_dimensions,
)


class TestCanonicalization:
    """Test input/output canonicalization to standard (T, I) / (T, O) format."""

    def test_canonicalize_input_1d_to_2d(self):
        """SISO input (T,) should become (T, 1)."""
        x = np.random.randn(1000)
        x_canon = canonicalize_input(x)

        assert x_canon.shape == (1000, 1)
        assert_array_equal(x_canon[:, 0], x)

    def test_canonicalize_input_already_2d(self):
        """MIMO input (T, I) should remain unchanged."""
        x = np.random.randn(1000, 3)
        x_canon = canonicalize_input(x)

        assert x_canon.shape == (1000, 3)
        assert_array_equal(x_canon, x)

    def test_canonicalize_input_invalid_3d(self):
        """3D input should raise ValueError."""
        x = np.random.randn(100, 3, 2)

        with pytest.raises(ValueError, match="Input must be 1D.*or 2D"):
            canonicalize_input(x)

    def test_canonicalize_output_1d_to_2d(self):
        """Single output (T,) should become (T, 1)."""
        y = np.random.randn(1000)
        y_canon = canonicalize_output(y)

        assert y_canon.shape == (1000, 1)
        assert_array_equal(y_canon[:, 0], y)

    def test_canonicalize_output_already_2d(self):
        """Multi-output (T, O) should remain unchanged."""
        y = np.random.randn(1000, 2)
        y_canon = canonicalize_output(y)

        assert y_canon.shape == (1000, 2)
        assert_array_equal(y_canon, y)

    def test_canonicalize_output_invalid_3d(self):
        """3D output should raise ValueError."""
        y = np.random.randn(100, 2, 3)

        with pytest.raises(ValueError, match="Output must be 1D.*or 2D"):
            canonicalize_output(y)

    def test_canonicalize_preserves_dtype(self):
        """Canonicalization should preserve data type."""
        x_f32 = np.random.randn(100).astype(np.float32)
        x_canon = canonicalize_input(x_f32)
        assert x_canon.dtype == np.float32

        y_f64 = np.random.randn(100).astype(np.float64)
        y_canon = canonicalize_output(y_f64)
        assert y_canon.dtype == np.float64


class TestValidation:
    """Test MIMO data validation."""

    def test_validate_siso_compatible(self):
        """SISO data (T,) x (T,) should pass validation."""
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        # Should not raise
        validate_mimo_data(x, y)

    def test_validate_mimo_compatible(self):
        """MIMO data (T, I) x (T, O) should pass validation."""
        x = np.random.randn(1000, 3)
        y = np.random.randn(1000, 2)

        # Should not raise
        validate_mimo_data(x, y)

    def test_validate_simo_compatible(self):
        """SIMO data (T,) x (T, O) should pass validation."""
        x = np.random.randn(1000)
        y = np.random.randn(1000, 3)

        # Should not raise
        validate_mimo_data(x, y)

    def test_validate_miso_compatible(self):
        """MISO data (T, I) x (T,) should pass validation."""
        x = np.random.randn(1000, 2)
        y = np.random.randn(1000)

        # Should not raise
        validate_mimo_data(x, y)

    def test_validate_mismatched_length(self):
        """Mismatched time lengths should raise ValueError."""
        x = np.random.randn(1000, 2)
        y = np.random.randn(500, 3)

        with pytest.raises(ValueError, match="same length"):
            validate_mimo_data(x, y)

    def test_validate_allow_different_length(self):
        """Different lengths allowed when require_same_length=False."""
        x = np.random.randn(1000, 2)
        y = np.random.randn(500, 3)

        # Should not raise
        validate_mimo_data(x, y, require_same_length=False)

    def test_validate_empty_input(self):
        """Empty input should raise ValueError."""
        x = np.array([])
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_mimo_data(x, y)

    def test_validate_empty_output(self):
        """Empty output should raise ValueError."""
        x = np.random.randn(100)
        y = np.array([])

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_mimo_data(x, y)

    def test_validate_invalid_input_shape(self):
        """3D input should raise ValueError."""
        x = np.random.randn(100, 2, 3)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="Input x must be 1D.*or 2D"):
            validate_mimo_data(x, y)

    def test_validate_invalid_output_shape(self):
        """3D output should raise ValueError."""
        x = np.random.randn(100)
        y = np.random.randn(100, 2, 3)

        with pytest.raises(ValueError, match="Output y must be 1D.*or 2D"):
            validate_mimo_data(x, y)


class TestDimensionInference:
    """Test dimension inference (T, I, O) from data."""

    def test_infer_siso_dimensions(self):
        """SISO (T,) x (T,) should infer I=1, O=1."""
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        T, I, O = infer_dimensions(x, y)

        assert T == 1000
        assert I == 1
        assert O == 1

    def test_infer_mimo_dimensions(self):
        """MIMO (T, I) x (T, O) should infer correct I, O."""
        x = np.random.randn(500, 3)
        y = np.random.randn(500, 2)

        T, I, O = infer_dimensions(x, y)

        assert T == 500
        assert I == 3
        assert O == 2

    def test_infer_simo_dimensions(self):
        """SIMO (T,) x (T, O) should infer I=1, O=O."""
        x = np.random.randn(800)
        y = np.random.randn(800, 4)

        T, I, O = infer_dimensions(x, y)

        assert T == 800
        assert I == 1
        assert O == 4

    def test_infer_miso_dimensions(self):
        """MISO (T, I) x (T,) should infer I=I, O=1."""
        x = np.random.randn(600, 5)
        y = np.random.randn(600)

        T, I, O = infer_dimensions(x, y)

        assert T == 600
        assert I == 5
        assert O == 1

    def test_infer_with_mismatched_length(self):
        """Mismatched lengths should raise ValueError."""
        x = np.random.randn(1000, 2)
        y = np.random.randn(500, 3)

        with pytest.raises(ValueError, match="same length"):
            infer_dimensions(x, y)

    def test_infer_with_invalid_shapes(self):
        """Invalid shapes should raise ValueError."""
        x = np.random.randn(100, 2, 3)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="Input x must be 1D.*or 2D"):
            infer_dimensions(x, y)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_sample_siso(self):
        """Single time sample SISO should work."""
        x = np.array([0.5])
        y = np.array([1.0])

        validate_mimo_data(x, y)
        T, I, O = infer_dimensions(x, y)

        assert T == 1
        assert I == 1
        assert O == 1

    def test_single_sample_mimo(self):
        """Single time sample MIMO should work."""
        x = np.array([[0.5, 0.3, 0.1]])  # (1, 3)
        y = np.array([[1.0, 2.0]])       # (1, 2)

        validate_mimo_data(x, y)
        T, I, O = infer_dimensions(x, y)

        assert T == 1
        assert I == 3
        assert O == 2

    def test_very_long_signal(self):
        """Very long signals should work."""
        T_long = 1_000_000
        x = np.random.randn(T_long, 2)
        y = np.random.randn(T_long, 3)

        validate_mimo_data(x, y)
        T, I, O = infer_dimensions(x, y)

        assert T == T_long
        assert I == 2
        assert O == 3

    def test_single_input_channel(self):
        """Single input channel (T, 1) should be valid MIMO."""
        x = np.random.randn(1000, 1)
        y = np.random.randn(1000, 3)

        validate_mimo_data(x, y)
        T, I, O = infer_dimensions(x, y)

        assert I == 1
        assert O == 3

    def test_single_output_channel(self):
        """Single output channel (T, 1) should be valid MIMO."""
        x = np.random.randn(1000, 3)
        y = np.random.randn(1000, 1)

        validate_mimo_data(x, y)
        T, I, O = infer_dimensions(x, y)

        assert I == 3
        assert O == 1
