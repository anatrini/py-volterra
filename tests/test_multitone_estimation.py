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


class TestMultiToneEstimation:
    """Test multi-tone kernel estimation accuracy."""

    @pytest.fixture
    def sample_rate(self):
        return 48000

    @pytest.fixture
    def known_kernel(self):
        """Create a known memoryless polynomial kernel for testing."""
        return VolterraKernelFull.from_polynomial_coeffs(
            N=256,  # Shorter for faster tests
            a1=1.0,
            a2=0.15,
            a3=0.03,
            a4=0.01,
            a5=0.02
        )

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

    def test_round_trip_linear_only(self, config_short, sample_rate):
        """Test estimation with linear-only kernel (h1)."""
        # Create linear-only kernel
        kernel = VolterraKernelFull.from_polynomial_coeffs(
            N=256,
            a1=1.0,  # Linear only
            a2=0.0,
            a3=0.0,
            a4=0.0,
            a5=0.0
        )

        # Generate excitation
        config_linear = MultiToneConfig(
            sample_rate=sample_rate,
            duration=1.0,
            num_tones=50,
            f_min=100.0,
            f_max=10000.0,
            max_order=1,  # Linear only
            amplitude=0.1
        )
        estimator = MultiToneEstimator(config_linear)
        excitation, frequencies = estimator.generate_excitation()

        # Process through known kernel
        processor = VolterraProcessorFull(kernel, sample_rate=sample_rate, use_numba=False)
        response = processor.process(excitation)

        # Estimate kernel
        estimated_kernel = estimator.estimate_kernel(
            excitation, response, frequencies, kernel_length=256
        )

        # Verify h1 matches (memoryless, so only first tap matters)
        # Linear system: very high accuracy expected
        assert_allclose(
            estimated_kernel.h1[0],
            kernel.h1[0],
            rtol=0.05,  # 5% tolerance
            err_msg="h1[0] mismatch in linear-only estimation"
        )

    def test_round_trip_with_nonlinearity(self, config_short, known_kernel, sample_rate):
        """
        Test full estimation with all orders.

        This is the critical test: known polynomial → process → estimate → verify.
        """
        # Generate excitation
        estimator = MultiToneEstimator(config_short)
        excitation, frequencies = estimator.generate_excitation()

        # Process through known kernel
        processor = VolterraProcessorFull(known_kernel, sample_rate=sample_rate, use_numba=False)
        response = processor.process(excitation)

        # Estimate kernel
        estimated_kernel = estimator.estimate_kernel(
            excitation, response, frequencies, kernel_length=256
        )

        # For memoryless polynomial, only tap [0] should be non-zero
        # Compare coefficients (with reasonable tolerance due to frequency-domain extraction)

        # h1 coefficient (linear)
        h1_known = known_kernel.h1[0]
        h1_estimated = estimated_kernel.h1[0]
        rel_error_h1 = abs(h1_estimated - h1_known) / abs(h1_known) if h1_known != 0 else 0
        assert rel_error_h1 < 0.1, f"h1 error {rel_error_h1*100:.1f}% > 10%"

        # h2 coefficient (quadratic)
        h2_known = known_kernel.h2[0] if known_kernel.h2 is not None else 0
        h2_estimated = estimated_kernel.h2[0] if estimated_kernel.h2 is not None else 0
        if h2_known != 0:
            rel_error_h2 = abs(h2_estimated - h2_known) / abs(h2_known)
            assert rel_error_h2 < 0.15, f"h2 error {rel_error_h2*100:.1f}% > 15%"

        # h3 coefficient (cubic)
        h3_known = known_kernel.h3_diagonal[0] if known_kernel.h3_diagonal is not None else 0
        h3_estimated = estimated_kernel.h3_diagonal[0] if estimated_kernel.h3_diagonal is not None else 0
        if h3_known != 0:
            rel_error_h3 = abs(h3_estimated - h3_known) / abs(h3_known)
            assert rel_error_h3 < 0.20, f"h3 error {rel_error_h3*100:.1f}% > 20%"

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
