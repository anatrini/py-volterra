"""
Tests for acoustic chain pipeline (Nonlinear → RIR).

These tests verify:
1. NonlinearThenRIR composition and processing
2. SISO and MIMO RIR configurations
3. RIR normalization and trimming
4. Configuration validation

Critical for STEP 5: Nonlinear+RIR Pipeline
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from volterra.pipelines import AcousticChainConfig, NonlinearThenRIR


class MockNonlinearModel:
    """Mock nonlinear model for testing."""

    def __init__(self, gain=1.0, is_fitted=True):
        self.gain = gain
        self.is_fitted = is_fitted

    def predict(self, x):
        """Simple gain + clipping."""
        if x.ndim == 1:
            return np.clip(self.gain * x, -1.0, 1.0)
        else:
            # Multi-channel: return first channel processed
            return np.clip(self.gain * x[:, 0], -1.0, 1.0)


class TestAcousticChainConfig:
    """Test AcousticChainConfig."""

    def test_config_defaults(self):
        """Default config should have sensible values."""
        config = AcousticChainConfig()

        assert config.rir_method == "fft"
        assert not config.normalize_rir
        assert config.trim_output

    def test_config_invalid_rir_method(self):
        """Invalid RIR method should raise error."""
        with pytest.raises(ValueError, match="rir_method"):
            AcousticChainConfig(rir_method="invalid")

    def test_config_custom_parameters(self):
        """Custom config parameters should be set."""
        config = AcousticChainConfig(rir_method="direct", normalize_rir=True, trim_output=False)

        assert config.rir_method == "direct"
        assert config.normalize_rir
        assert not config.trim_output


class TestNonlinearThenRIRInitialization:
    """Test NonlinearThenRIR initialization."""

    def test_initialization_siso(self):
        """Basic SISO initialization should work."""
        nl_model = MockNonlinearModel()
        rir = np.random.randn(100)

        chain = NonlinearThenRIR(nl_model, rir, sample_rate=48000)

        assert chain.rir_length == 100
        assert chain.n_rir_channels == 1
        assert chain.sample_rate == 48000
        assert not chain.is_mimo

    def test_initialization_mimo(self):
        """MIMO (stereo) initialization should work."""
        nl_model = MockNonlinearModel()
        rir_stereo = np.random.randn(200, 2)

        chain = NonlinearThenRIR(nl_model, rir_stereo, sample_rate=48000)

        assert chain.rir_length == 200
        assert chain.n_rir_channels == 2
        assert chain.is_mimo

    def test_initialization_with_config(self):
        """Initialization with custom config."""
        nl_model = MockNonlinearModel()
        rir = np.random.randn(100)
        config = AcousticChainConfig(normalize_rir=True, trim_output=False)

        chain = NonlinearThenRIR(nl_model, rir, config=config)

        assert chain.config.normalize_rir
        assert not chain.config.trim_output

    def test_initialization_invalid_no_predict_method(self):
        """Model without predict() should raise error."""

        class BadModel:
            pass

        rir = np.random.randn(100)

        with pytest.raises(ValueError, match="predict"):
            NonlinearThenRIR(BadModel(), rir)

    def test_initialization_invalid_not_fitted(self):
        """Unfitted model should raise error."""
        nl_model = MockNonlinearModel(is_fitted=False)
        rir = np.random.randn(100)

        with pytest.raises(ValueError, match="fitted"):
            NonlinearThenRIR(nl_model, rir)

    def test_initialization_invalid_rir_3d(self):
        """3D RIR should raise error."""
        nl_model = MockNonlinearModel()
        rir_3d = np.random.randn(100, 2, 3)

        with pytest.raises(ValueError, match="1D or 2D"):
            NonlinearThenRIR(nl_model, rir_3d)

    def test_initialization_empty_rir(self):
        """Empty RIR should raise error."""
        nl_model = MockNonlinearModel()
        rir_empty = np.array([])

        with pytest.raises(ValueError, match="empty"):
            NonlinearThenRIR(nl_model, rir_empty)

    def test_repr(self):
        """String representation should show components."""
        nl_model = MockNonlinearModel()
        rir = np.random.randn(100)

        chain = NonlinearThenRIR(nl_model, rir, sample_rate=44100)
        repr_str = repr(chain)

        assert "MockNonlinearModel" in repr_str
        assert "rir_length=100" in repr_str
        assert "44100Hz" in repr_str


class TestNonlinearThenRIRProcessing:
    """Test NonlinearThenRIR processing."""

    def test_process_siso_basic(self):
        """Basic SISO processing should work."""
        nl_model = MockNonlinearModel(gain=1.0)
        rir = np.array([1.0, 0.5, 0.25])  # Simple decaying RIR

        chain = NonlinearThenRIR(nl_model, rir)

        x = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        y = chain.process(x)

        # Output should be 1D for SISO
        assert y.ndim == 1

    def test_process_mimo_stereo(self):
        """Stereo RIR processing should work."""
        nl_model = MockNonlinearModel(gain=1.0)
        rir_stereo = np.random.randn(50, 2)

        chain = NonlinearThenRIR(nl_model, rir_stereo)

        x = np.random.randn(100)
        y = chain.process(x)

        # Output should be 2D for MIMO
        assert y.ndim == 2
        assert y.shape[1] == 2  # Stereo

    def test_process_impulse_response(self):
        """Impulse input should give nonlinearity → RIR."""
        nl_model = MockNonlinearModel(gain=2.0)
        rir = np.array([1.0, 0.5, 0.25, 0.125])

        config = AcousticChainConfig(trim_output=False)
        chain = NonlinearThenRIR(nl_model, rir, config=config)

        # Impulse input
        x = np.zeros(10)
        x[0] = 0.5  # Impulse at t=0

        y = chain.process(x)

        # Expected: nonlinear output is 2.0 * 0.5 = 1.0 at t=0
        # Then convolved with RIR
        # First few samples should match RIR scaled by nonlinear gain
        assert_allclose(y[0], 1.0, rtol=1e-5)
        assert_allclose(y[1], 0.5, rtol=1e-5)
        assert_allclose(y[2], 0.25, rtol=1e-5)

    def test_process_with_trimming(self):
        """trim_output=True should maintain input length."""
        nl_model = MockNonlinearModel()
        rir = np.random.randn(50)

        config = AcousticChainConfig(trim_output=True)
        chain = NonlinearThenRIR(nl_model, rir, config=config)

        x = np.random.randn(100)
        y = chain.process(x)

        # Output should NOT be longer than intermediate signal after nonlinearity
        # (accounting for memory in nonlinear model)
        assert len(y) <= len(x)

    def test_process_without_trimming(self):
        """trim_output=False should include full convolution."""
        nl_model = MockNonlinearModel()
        rir = np.random.randn(50)

        config = AcousticChainConfig(trim_output=False)
        chain = NonlinearThenRIR(nl_model, rir, config=config)

        x = np.random.randn(100)
        y = chain.process(x)

        # Full convolution should be longer
        # Exact length depends on nonlinear model output length
        # Just check it exists and is reasonable
        assert len(y) > 0

    def test_process_fft_vs_direct(self):
        """FFT and direct convolution should give same results."""
        nl_model = MockNonlinearModel()
        rir = np.random.randn(30)
        x = np.random.randn(100)

        # FFT convolution
        config_fft = AcousticChainConfig(rir_method="fft", trim_output=True)
        chain_fft = NonlinearThenRIR(nl_model, rir, config=config_fft)
        y_fft = chain_fft.process(x)

        # Direct convolution
        config_direct = AcousticChainConfig(rir_method="direct", trim_output=True)
        chain_direct = NonlinearThenRIR(nl_model, rir, config=config_direct)
        y_direct = chain_direct.process(x)

        # Should be very close
        assert_allclose(y_fft, y_direct, rtol=1e-10)


class TestNonlinearThenRIRNormalization:
    """Test RIR normalization."""

    def test_normalize_rir_enabled(self):
        """normalize_rir=True should normalize RIR energy."""
        nl_model = MockNonlinearModel()
        rir = np.array([1.0, 2.0, 3.0, 4.0])

        config = AcousticChainConfig(normalize_rir=True)
        chain = NonlinearThenRIR(nl_model, rir, config=config)

        # Check RIR is normalized to unit energy
        rir_energy = np.sum(chain.rir[:, 0] ** 2)
        assert_allclose(rir_energy, 1.0, rtol=1e-10)

    def test_normalize_rir_disabled(self):
        """normalize_rir=False should preserve RIR."""
        nl_model = MockNonlinearModel()
        rir = np.array([1.0, 2.0, 3.0, 4.0])

        config = AcousticChainConfig(normalize_rir=False)
        chain = NonlinearThenRIR(nl_model, rir, config=config)

        # RIR should be unchanged
        assert_allclose(chain.rir[:, 0], rir)

    def test_normalize_rir_stereo(self):
        """Normalization should work per-channel for stereo."""
        nl_model = MockNonlinearModel()
        rir_stereo = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]])

        config = AcousticChainConfig(normalize_rir=True)
        chain = NonlinearThenRIR(nl_model, rir_stereo, config=config)

        # Each channel should have unit energy
        for ch in range(2):
            energy = np.sum(chain.rir[:, ch] ** 2)
            assert_allclose(energy, 1.0, rtol=1e-10)


class TestNonlinearThenRIRAccessors:
    """Test accessor methods."""

    def test_get_rir_siso(self):
        """get_rir() should return RIR for SISO."""
        nl_model = MockNonlinearModel()
        rir = np.array([1.0, 0.5, 0.25])

        chain = NonlinearThenRIR(nl_model, rir)
        rir_ch = chain.get_rir(channel=0)

        assert_allclose(rir_ch, rir)

    def test_get_rir_stereo(self):
        """get_rir() should return correct channel for stereo."""
        nl_model = MockNonlinearModel()
        rir_stereo = np.array([[1.0, 2.0], [0.5, 1.5], [0.25, 1.0]])

        chain = NonlinearThenRIR(nl_model, rir_stereo)

        rir_left = chain.get_rir(channel=0)
        rir_right = chain.get_rir(channel=1)

        assert_allclose(rir_left, rir_stereo[:, 0])
        assert_allclose(rir_right, rir_stereo[:, 1])

    def test_get_rir_invalid_channel(self):
        """get_rir() with invalid channel should raise error."""
        nl_model = MockNonlinearModel()
        rir = np.random.randn(100)

        chain = NonlinearThenRIR(nl_model, rir)

        with pytest.raises(ValueError, match="channel"):
            chain.get_rir(channel=5)

    def test_get_nonlinear_model(self):
        """get_nonlinear_model() should return the model."""
        nl_model = MockNonlinearModel()
        rir = np.random.randn(100)

        chain = NonlinearThenRIR(nl_model, rir)
        retrieved_model = chain.get_nonlinear_model()

        assert retrieved_model is nl_model


class TestNonlinearThenRIREdgeCases:
    """Test edge cases."""

    def test_very_long_rir(self):
        """Very long RIR should work."""
        nl_model = MockNonlinearModel()
        rir = np.random.randn(10000)  # 10k samples

        chain = NonlinearThenRIR(nl_model, rir)
        x = np.random.randn(1000)
        y = chain.process(x)

        assert len(y) > 0

    def test_single_sample_rir(self):
        """Single-sample RIR (Dirac delta) should work."""
        nl_model = MockNonlinearModel(gain=2.0)
        rir = np.array([1.0])  # Dirac delta

        chain = NonlinearThenRIR(nl_model, rir)
        x = np.array([0.5, 0.3, 0.1])
        y = chain.process(x)

        # With Dirac RIR, output should match nonlinear model output
        expected = nl_model.predict(x)
        assert_allclose(y, expected, rtol=1e-10)

    def test_short_input_signal(self):
        """Short input signal should work."""
        nl_model = MockNonlinearModel()
        rir = np.random.randn(50)

        chain = NonlinearThenRIR(nl_model, rir)
        x = np.array([1.0])  # Single sample
        y = chain.process(x)

        assert len(y) > 0

    def test_multi_channel_rir(self):
        """Multi-channel RIR (more than 2) should work."""
        nl_model = MockNonlinearModel()
        rir_5ch = np.random.randn(100, 5)  # 5 channels

        chain = NonlinearThenRIR(nl_model, rir_5ch)
        x = np.random.randn(200)
        y = chain.process(x)

        assert y.shape[1] == 5  # 5 output channels
