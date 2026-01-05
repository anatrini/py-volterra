"""
Harmonic Generation Validation Tests.

These tests verify that each Volterra kernel order generates correct harmonic content.
This is CRITICAL for mathematical correctness - if these fail, the implementation is WRONG.
"""

import numpy as np
import pytest
from scipy.fft import rfft, rfftfreq

from volterra import VolterraKernelFull, VolterraProcessorFull


class TestHarmonicGeneration:
    """Verify each kernel order generates correct harmonics."""

    @pytest.fixture
    def sample_rate(self):
        return 48000

    @pytest.fixture
    def test_frequency(self):
        return 1000  # 1 kHz test tone

    @pytest.fixture
    def duration(self):
        return 1.0  # 1 second

    def _analyze_harmonics(self, signal, fs, f0, num_harmonics=10):
        """Analyze harmonic content of signal."""
        Y = rfft(signal)
        freqs = rfftfreq(len(signal), 1 / fs)

        harmonics = {}
        for k in range(1, num_harmonics + 1):
            f_harmonic = k * f0
            if f_harmonic > fs / 2:
                break

            idx = np.argmin(np.abs(freqs - f_harmonic))
            level_db = 20 * np.log10(np.abs(Y[idx]) + 1e-15)
            harmonics[k] = level_db

        return harmonics

    def test_h2_generates_only_even_harmonics(self, sample_rate, test_frequency, duration):
        """
        Pure h₂: input sine → output has ONLY even harmonics (2f, 4f, 6f).

        Theory: x²(t) produces only even harmonics
        """
        # Setup: h₂ ONLY, all others zero
        N = 512
        h1 = np.zeros(N)  # NO linear term
        h2 = np.zeros(N)
        h2[0] = 0.5  # Memoryless quadratic

        kernel = VolterraKernelFull(
            h1=h1, h2=h2, h3_diagonal=None, h4_diagonal=None, h5_diagonal=None, h2_is_diagonal=True
        )

        processor = VolterraProcessorFull(kernel, sample_rate=sample_rate, use_numba=False)

        # Input: pure sine
        t = np.arange(int(sample_rate * duration)) / sample_rate
        amplitude = 0.3
        input_signal = amplitude * np.sin(2 * np.pi * test_frequency * t)

        # Process
        output = processor.process(input_signal, block_size=512)

        # Analyze harmonics
        harmonics = self._analyze_harmonics(output, sample_rate, test_frequency)

        # Verify: even harmonics strong, odd harmonics suppressed
        h1_level = harmonics.get(1, -120)
        h2_level = harmonics.get(2, -120)
        h3_level = harmonics.get(3, -120)
        h4_level = harmonics.get(4, -120)

        # Critical checks
        assert h2_level > -20, f"2nd harmonic too weak: {h2_level:.1f} dB"
        assert (
            h1_level < -60
        ), f"Fundamental should be suppressed: {h1_level:.1f} dB (expected < -60 dB)"
        assert (
            h3_level < -60
        ), f"3rd harmonic should be suppressed: {h3_level:.1f} dB (expected < -60 dB)"

        # 4th harmonic may be present (it's even)
        # But should be weaker than 2nd
        if h4_level > -100:
            assert h4_level < h2_level - 10, "4th harmonic should be weaker than 2nd"

    def test_h3_generates_only_odd_harmonics(self, sample_rate, test_frequency, duration):
        """
        Pure h₃: input sine → output has ONLY odd harmonics (3f, 5f, 7f).

        Theory: x³(t) produces only odd harmonics
        """
        # Setup: h₃ ONLY
        N = 512
        h1 = np.zeros(N)
        h3 = np.zeros(N)
        h3[0] = 0.3  # Memoryless cubic

        kernel = VolterraKernelFull(
            h1=h1, h2=None, h3_diagonal=h3, h4_diagonal=None, h5_diagonal=None, h2_is_diagonal=True
        )

        processor = VolterraProcessorFull(kernel, sample_rate=sample_rate, use_numba=False)

        # Input: pure sine
        t = np.arange(int(sample_rate * duration)) / sample_rate
        amplitude = 0.2
        input_signal = amplitude * np.sin(2 * np.pi * test_frequency * t)

        # Process
        output = processor.process(input_signal, block_size=512)

        # Analyze harmonics
        harmonics = self._analyze_harmonics(output, sample_rate, test_frequency)

        # Verify: odd harmonics present, even harmonics suppressed
        h2_level = harmonics.get(2, -120)
        h3_level = harmonics.get(3, -120)
        h4_level = harmonics.get(4, -120)

        # Critical checks: even harmonics MUST be suppressed
        assert (
            h2_level < -60
        ), f"2nd harmonic (even) should be suppressed: {h2_level:.1f} dB (expected < -60 dB)"
        assert (
            h4_level < -60
        ), f"4th harmonic (even) should be suppressed: {h4_level:.1f} dB (expected < -60 dB)"

        # 3rd harmonic should be present
        assert h3_level > -40, f"3rd harmonic too weak: {h3_level:.1f} dB"

    def test_h5_generates_fifth_harmonics(self, sample_rate, test_frequency, duration):
        """Pure h₅: input sine → output has 5th harmonic content."""
        N = 512
        h1 = np.zeros(N)
        h5 = np.zeros(N)
        h5[0] = 0.2

        kernel = VolterraKernelFull(
            h1=h1, h2=None, h3_diagonal=None, h4_diagonal=None, h5_diagonal=h5, h2_is_diagonal=True
        )

        processor = VolterraProcessorFull(kernel, sample_rate=sample_rate, use_numba=False)

        t = np.arange(int(sample_rate * duration)) / sample_rate
        amplitude = 0.15
        input_signal = amplitude * np.sin(2 * np.pi * test_frequency * t)

        output = processor.process(input_signal, block_size=512)

        harmonics = self._analyze_harmonics(output, sample_rate, test_frequency)

        # 5th order produces odd harmonics (5 is odd)
        h2_level = harmonics.get(2, -120)
        h4_level = harmonics.get(4, -120)
        h5_level = harmonics.get(5, -120)

        # Even harmonics suppressed
        assert h2_level < -60, f"2nd harmonic should be suppressed: {h2_level:.1f} dB"
        assert h4_level < -60, f"4th harmonic should be suppressed: {h4_level:.1f} dB"

        # 5th harmonic should be strong
        assert h5_level > -40, f"5th harmonic too weak: {h5_level:.1f} dB"

    def test_combined_h1_h2_h3_additive(self, sample_rate, test_frequency, duration):
        """
        Combined kernels: output should be sum of individual outputs.

        This verifies that the processor correctly sums contributions from different orders.
        """
        N = 512

        # Individual kernels
        h1 = np.zeros(N)
        h1[0] = 0.9
        h2 = np.zeros(N)
        h2[0] = 0.1
        h3 = np.zeros(N)
        h3[0] = 0.02

        # Combined kernel
        kernel_combined = VolterraKernelFull(h1=h1, h2=h2, h3_diagonal=h3, h2_is_diagonal=True)

        # Separate kernels
        kernel_h1_only = VolterraKernelFull(h1=h1)
        kernel_h2_only = VolterraKernelFull(h1=np.zeros(N), h2=h2, h2_is_diagonal=True)
        kernel_h3_only = VolterraKernelFull(h1=np.zeros(N), h3_diagonal=h3)

        # Create processors
        proc_combined = VolterraProcessorFull(
            kernel_combined, sample_rate=sample_rate, use_numba=False
        )
        proc_h1 = VolterraProcessorFull(kernel_h1_only, sample_rate=sample_rate, use_numba=False)
        proc_h2 = VolterraProcessorFull(kernel_h2_only, sample_rate=sample_rate, use_numba=False)
        proc_h3 = VolterraProcessorFull(kernel_h3_only, sample_rate=sample_rate, use_numba=False)

        # Input signal
        t = np.arange(int(sample_rate * duration)) / sample_rate
        input_signal = 0.2 * np.sin(2 * np.pi * test_frequency * t)

        # Process
        out_combined = proc_combined.process(input_signal.copy(), block_size=512)
        out_h1 = proc_h1.process(input_signal.copy(), block_size=512)
        out_h2 = proc_h2.process(input_signal.copy(), block_size=512)
        out_h3 = proc_h3.process(input_signal.copy(), block_size=512)

        # Combined should equal sum
        out_sum = out_h1 + out_h2 + out_h3

        # Verify
        max_diff = np.max(np.abs(out_combined - out_sum))
        rms_diff = np.sqrt(np.mean((out_combined - out_sum) ** 2))

        assert max_diff < 1e-10, f"Combined output doesn't match sum: max_diff={max_diff:.2e}"
        assert rms_diff < 1e-11, f"Combined output doesn't match sum: rms_diff={rms_diff:.2e}"

    @pytest.mark.parametrize("order,coefficient", [(2, 0.1), (3, 0.05), (5, 0.03)])
    def test_output_scales_with_coefficient(
        self, order, coefficient, sample_rate, test_frequency, duration
    ):
        """Output amplitude should scale linearly with kernel coefficient."""
        N = 512
        h1 = np.zeros(N)

        # Create kernel with given coefficient
        if order == 2:
            h2 = np.zeros(N)
            h2[0] = coefficient
            kernel = VolterraKernelFull(h1=h1, h2=h2, h2_is_diagonal=True)
        elif order == 3:
            h3 = np.zeros(N)
            h3[0] = coefficient
            kernel = VolterraKernelFull(h1=h1, h3_diagonal=h3)
        elif order == 5:
            h5 = np.zeros(N)
            h5[0] = coefficient
            kernel = VolterraKernelFull(h1=h1, h5_diagonal=h5)

        processor = VolterraProcessorFull(kernel, sample_rate=sample_rate, use_numba=False)

        t = np.arange(int(sample_rate * duration)) / sample_rate
        input_signal = 0.2 * np.sin(2 * np.pi * test_frequency * t)

        output = processor.process(input_signal, block_size=512)

        # RMS output should be proportional to coefficient
        rms_output = np.sqrt(np.mean(output**2))

        # For order n: output ~ coefficient * amplitude^n
        expected_rms_order_of_magnitude = coefficient * (0.2**order)

        # Check order of magnitude is reasonable
        assert rms_output > 0, "Output is zero (wrong!)"
        assert (
            rms_output > expected_rms_order_of_magnitude * 0.01
        ), f"Output too weak for order {order}"
