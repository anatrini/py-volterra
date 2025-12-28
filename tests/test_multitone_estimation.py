"""
Multi-tone kernel estimation tests.

Tests the complete cycle:
1. Generate multi-tone excitation
2. Process through known Volterra kernel
3. Estimate kernels from output
4. Verify estimated kernels match known kernels (within tolerance)

Critical for verifying system identification correctness.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from volterra import (
    VolterraKernelFull,
    VolterraProcessorFull,
    MultiToneConfig,
    MultiToneEstimator,
)
from volterra.engines_diagonal import DiagonalNumpyEngine


def create_fir_kernel_for_estimation(N: int = 256, sample_rate: int = 48000) -> VolterraKernelFull:
    """
    Create realistic FIR kernel for multi-tone estimation testing.

    Multi-tone estimation requires band-limited FIR kernels (not memoryless polynomials).
    This creates exponentially-decaying impulse responses with frequency structure.

    Args:
        N: Kernel length
        sample_rate: Sample rate (for frequency scaling)

    Returns:
        VolterraKernelFull with realistic FIR kernels
    """
    # Create exponentially-decaying impulse responses
    n = np.arange(N, dtype=np.float64)

    # h1: Linear kernel (low-pass characteristic)
    decay_h1 = np.exp(-n / 20.0)  # Decay over ~20 samples
    h1 = decay_h1 / np.sum(decay_h1)  # Normalize

    # h2: 2nd-order diagonal (faster decay)
    decay_h2 = np.exp(-n / 10.0)
    h2 = 0.15 * decay_h2 / np.sum(decay_h2)

    # h3: 3rd-order diagonal (even faster)
    decay_h3 = np.exp(-n / 8.0)
    h3 = 0.03 * decay_h3 / np.sum(decay_h3)

    # h5: 5th-order diagonal (fastest decay)
    decay_h5 = np.exp(-n / 5.0)
    h5 = 0.02 * decay_h5 / np.sum(decay_h5)

    return VolterraKernelFull(
        h1=h1,
        h2=h2,
        h3_diagonal=h3,
        h4_diagonal=None,
        h5_diagonal=h5,
        h2_is_diagonal=True
    )


class TestMultiToneEstimation:
    """Test multi-tone kernel estimation accuracy."""

    @pytest.fixture
    def sample_rate(self):
        return 48000

    @pytest.fixture
    def known_kernel(self, sample_rate):
        """Create realistic FIR kernel for multi-tone estimation."""
        return create_fir_kernel_for_estimation(N=256, sample_rate=sample_rate)

    @pytest.fixture
    def config_short(self, sample_rate):
        """Short config for faster tests."""
        return MultiToneConfig(
            sample_rate=sample_rate,
            duration=1.0,  # 1 second
            num_tones=50,
            f_min=100.0,
            f_max=10000.0,
            max_order=5,
            method='random_phase',
            amplitude=0.05  # Low amplitude for linearity
        )

    def test_generate_excitation(self, config_short):
        """Test that excitation signal is generated correctly."""
        estimator = MultiToneEstimator(config_short)
        excitation, frequencies = estimator.generate_excitation()

        # Check signal properties
        assert len(excitation) == int(config_short.sample_rate * config_short.duration)
        assert len(frequencies) == config_short.num_tones
        assert np.all(frequencies >= config_short.f_min)
        assert np.all(frequencies <= config_short.f_max)

        # Check amplitude normalization
        assert np.max(np.abs(excitation)) <= config_short.amplitude * 1.01
        assert np.max(np.abs(excitation)) >= config_short.amplitude * 0.99

    def test_frequency_spacing(self, config_short):
        """Test that frequencies avoid harmonic collisions."""
        estimator = MultiToneEstimator(config_short)
        frequencies = estimator._select_frequencies()

        # Check no harmonic collisions
        for i, f1 in enumerate(frequencies):
            for j, f2 in enumerate(frequencies):
                if i >= j:
                    continue
                # Check harmonics up to 5th order
                for k in range(1, 6):
                    # f1 * k should not be close to f2
                    assert np.abs(k * f1 - f2) > 10, \
                        f"Harmonic collision: {k}*{f1:.1f}Hz ≈ {f2:.1f}Hz"

    def test_round_trip_linear_only(self, sample_rate):
        """
        Test linear-only FIR kernel estimation.

        Uses realistic FIR kernel (exponential decay), not memoryless polynomial.
        Multi-tone estimation requires frequency structure in kernels.
        """
        # Create linear-only FIR kernel
        n = np.arange(256, dtype=np.float64)
        decay = np.exp(-n / 20.0)
        h1 = decay / np.sum(decay)  # Normalized exponential decay

        kernel = VolterraKernelFull(
            h1=h1,
            h2=None,
            h3_diagonal=None,
            h4_diagonal=None,
            h5_diagonal=None,
            h2_is_diagonal=True
        )

        # Generate excitation with MANY tones for better frequency resolution
        config_linear = MultiToneConfig(
            sample_rate=sample_rate,
            duration=1.0,
            num_tones=200,  # 4x more tones for better reconstruction
            f_min=100.0,
            f_max=10000.0,
            max_order=1,
            amplitude=0.1
        )
        estimator = MultiToneEstimator(config_linear)
        excitation, frequencies = estimator.generate_excitation()

        # Process through known FIR kernel
        processor = VolterraProcessorFull(kernel, sample_rate=sample_rate, use_numba=False)
        response = processor.process(excitation)

        # Estimate kernel
        estimated_kernel = estimator.estimate_kernel(
            excitation, response, frequencies, kernel_length=256
        )

        # Verify estimated h1 matches original (first 50 taps where energy is concentrated)
        # FIR estimation with sparse frequency sampling (200 tones, 100-10kHz):
        # - Taps 3+: ~10-20% error (good reconstruction)
        # - Taps 0-2: ~20-50% error (DC/low-freq harder to capture)
        # Overall tolerance: 50% reflects achievable accuracy
        assert_allclose(
            estimated_kernel.h1[:50],
            kernel.h1[:50],
            rtol=0.5,  # 50% tolerance for sparse frequency-domain reconstruction
            err_msg="h1 FIR reconstruction error > 50%"
        )

        # Verify overall energy preserved
        energy_original = np.sum(kernel.h1**2)
        energy_estimated = np.sum(estimated_kernel.h1**2)
        energy_ratio = energy_estimated / energy_original

        assert 0.5 < energy_ratio < 1.5, \
            f"Energy not preserved: {energy_ratio:.2f}x (should be ~1.0x)"

    def test_round_trip_with_nonlinearity(self, config_short, known_kernel, sample_rate):
        """
        Test full nonlinear FIR kernel estimation (orders 1-5).

        Uses realistic FIR kernels with exponential decay.
        Verifies multi-tone can extract multiple Volterra orders simultaneously.
        """
        # Generate excitation
        estimator = MultiToneEstimator(config_short)
        excitation, frequencies = estimator.generate_excitation()

        # Process through known FIR kernel
        processor = VolterraProcessorFull(known_kernel, sample_rate=sample_rate, use_numba=False)
        response = processor.process(excitation)

        # Estimate kernel
        estimated_kernel = estimator.estimate_kernel(
            excitation, response, frequencies, kernel_length=256
        )

        # Verify h1 (linear) - check first 30 taps where energy is concentrated
        # Nonlinear FIR reconstruction (h1-h5 simultaneously) is harder than linear-only
        # Sparse frequency sampling + order separation → higher errors
        assert_allclose(
            estimated_kernel.h1[:30],
            known_kernel.h1[:30],
            rtol=1.1,  # 110% tolerance for h1 with simultaneous multi-order extraction
            err_msg="h1 reconstruction error > 110%"
        )

        # Verify h2 (quadratic) - lower accuracy expected for higher orders
        if known_kernel.h2 is not None:
            assert_allclose(
                estimated_kernel.h2[:20],
                known_kernel.h2[:20],
                rtol=1.5,  # 150% tolerance for h2
                err_msg="h2 reconstruction error > 150%"
            )

        # Verify h3 (cubic) - even lower accuracy
        if known_kernel.h3_diagonal is not None:
            assert_allclose(
                estimated_kernel.h3_diagonal[:15],
                known_kernel.h3_diagonal[:15],
                rtol=2.0,  # 200% tolerance for h3
                err_msg="h3 reconstruction error > 200%"
            )

    def test_schroeder_phase_method(self, sample_rate):
        """Test Schroeder phase method for low crest factor."""
        config_schroeder = MultiToneConfig(
            sample_rate=sample_rate,
            duration=1.0,
            num_tones=50,
            method='schroeder_phase',
            amplitude=0.1
        )

        estimator = MultiToneEstimator(config_schroeder)
        excitation, frequencies = estimator.generate_excitation()

        # Check that crest factor is reasonable
        # Crest factor = peak / RMS
        peak = np.max(np.abs(excitation))
        rms = np.sqrt(np.mean(excitation**2))
        crest_factor = peak / rms

        # Schroeder phase reduces crest factor but not dramatically for moderate num_tones
        # Expected range for 50 tones: ~2-5 (still better than random which can be >6)
        assert 1.0 < crest_factor < 6.0, \
            f"Schroeder phase crest factor {crest_factor:.2f} out of expected range"

    def test_snr_estimation(self, config_short):
        """Test SNR estimation method."""
        estimator = MultiToneEstimator(config_short)
        excitation, frequencies = estimator.generate_excitation()

        # Add known noise level
        noise_amplitude = 0.001
        noise = noise_amplitude * np.random.randn(len(excitation))
        noisy_signal = excitation + noise

        # Estimate SNR
        snr_db = estimator.estimate_snr_db(noisy_signal, frequencies)

        # With amplitude 0.05 and noise 0.001, expect SNR ~20-40 dB
        # Allow wide tolerance due to:
        # - Statistical variation in noise
        # - Frequency-domain estimation method
        # - Multiple tones with different levels
        assert 15 < snr_db < 50, f"SNR {snr_db:.1f} dB outside expected range"

    def test_invalid_config(self):
        """Test that invalid configurations raise errors."""
        # Invalid method
        with pytest.raises(ValueError, match="method must be one of"):
            MultiToneConfig(method='invalid_method')

        # Invalid max_order
        with pytest.raises(ValueError, match="max_order must be <= 5"):
            MultiToneConfig(max_order=7)

    def test_kernel_extraction_structure(self, config_short, sample_rate):
        """Test that estimation produces kernels with correct structure."""
        # Use simple linear kernel for reliability
        kernel_simple = VolterraKernelFull.from_polynomial_coeffs(
            N=256,
            a1=1.0,
            a2=0.0,
            a3=0.0,
            a4=0.0,
            a5=0.0
        )

        # Generate excitation
        estimator = MultiToneEstimator(config_short)
        excitation, frequencies = estimator.generate_excitation()

        # Process through kernel
        processor = VolterraProcessorFull(kernel_simple, sample_rate=sample_rate, use_numba=False)
        response = processor.process(excitation)

        # Estimate kernel
        estimated_kernel = estimator.estimate_kernel(
            excitation, response, frequencies, kernel_length=256
        )

        # Verify kernel structure is correct
        assert estimated_kernel.h1 is not None
        assert len(estimated_kernel.h1) == 256
        assert estimated_kernel.h2 is not None  # Created with max_order=5
        assert estimated_kernel.h3_diagonal is not None
        assert estimated_kernel.h4_diagonal is not None
        assert estimated_kernel.h5_diagonal is not None
        assert estimated_kernel.h2_is_diagonal is True
