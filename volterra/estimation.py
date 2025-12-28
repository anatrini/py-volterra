"""
Multi-tone Volterra kernel estimation.

This module implements frequency-domain kernel extraction using carefully
designed multi-tone excitation signals. The method separates different
Volterra orders through harmonic content analysis.

Theory:
-------
For a multi-tone signal x(t) = Σ Aₖ·cos(2πfₖt + φₖ):
- Linear (h1): appears at fundamental frequencies fₖ
- 2nd-order diagonal: appears at 2fₖ
- 3rd-order diagonal: appears at 3fₖ
- 5th-order diagonal: appears at 5fₖ

Frequency-domain relationships:
- Y(2f) = (1/2) · H2_diag(f) · |X(f)|²
- Y(3f) = (1/6) · H3_diag(f) · |X(f)|³
- Y(5f) = (1/120) · H5_diag(f) · |X(f)|⁵

References:
-----------
- Reed & Hawksford (1996) "Identification of discrete Volterra series"
- Novák et al. (2015) "Nonlinear system identification using exponential swept-sine signal"
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
from scipy import fft
from volterra.kernels_full import VolterraKernelFull, ArrayF


@dataclass
class MultiToneConfig:
    """Configuration for multi-tone kernel estimation."""

    sample_rate: int = 48000
    duration: float = 2.0
    num_tones: int = 100
    f_min: float = 20.0
    f_max: float = 20000.0
    max_order: int = 5
    method: str = 'random_phase'  # 'random_phase' or 'schroeder_phase'
    amplitude: float = 0.1  # Peak amplitude to avoid clipping

    def __post_init__(self):
        valid_methods = ['random_phase', 'schroeder_phase']
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        if self.max_order > 5:
            raise ValueError("max_order must be <= 5")


class MultiToneEstimator:
    """
    Extract Volterra kernels using multi-tone excitation.

    This method provides better SNR and control over frequency content
    compared to swept-sine, especially for higher orders.

    Example:
        >>> config = MultiToneConfig(num_tones=100, max_order=5)
        >>> estimator = MultiToneEstimator(config)
        >>> excitation, freqs = estimator.generate_excitation()
        >>> # Send excitation to system, record response
        >>> kernel = estimator.estimate_kernel(excitation, response, freqs)
    """

    def __init__(self, config: MultiToneConfig):
        self.config = config
        self.rng = np.random.RandomState(42)  # Reproducible random phases

    def generate_excitation(self) -> Tuple[ArrayF, ArrayF]:
        """
        Generate multi-tone excitation signal with optimal frequency spacing.

        Returns:
            excitation: Time-domain signal
            frequencies: Tone frequencies used
        """
        # Select frequencies avoiding harmonic overlap
        frequencies = self._select_frequencies()

        # Generate phases
        if self.config.method == 'random_phase':
            phases = self.rng.uniform(0, 2*np.pi, len(frequencies))
        else:  # schroeder_phase
            phases = self._schroeder_phases(len(frequencies))

        # Build time-domain signal
        t = np.arange(int(self.config.sample_rate * self.config.duration)) / self.config.sample_rate
        excitation = np.zeros_like(t, dtype=np.float64)

        for f, phi in zip(frequencies, phases):
            excitation += np.cos(2 * np.pi * f * t + phi)

        # Normalize to target amplitude
        excitation *= self.config.amplitude / np.max(np.abs(excitation))

        return excitation, frequencies

    def _select_frequencies(self) -> ArrayF:
        """
        Select frequencies avoiding harmonic collisions.

        For order-k Volterra, avoid: f_i * m = f_j for m <= k
        Uses prime-based spacing for optimal separation.
        """
        f_min = self.config.f_min
        f_max = self.config.f_max
        num_tones = self.config.num_tones
        max_order = self.config.max_order

        # Logarithmic spacing as baseline
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), num_tones * 3)

        # Filter to avoid harmonics up to max_order
        selected = []
        for f in freqs:
            # Check if this frequency or its low harmonics conflict
            conflict = False
            for existing in selected:
                for k in range(1, max_order + 1):
                    if np.abs(k * existing - f) < 10:  # 10 Hz tolerance
                        conflict = True
                        break
                    if np.abs(k * f - existing) < 10:
                        conflict = True
                        break
                if conflict:
                    break

            if not conflict:
                selected.append(f)

            if len(selected) >= num_tones:
                break

        return np.array(selected[:num_tones])

    def _schroeder_phases(self, N: int) -> ArrayF:
        """
        Schroeder-phase multisine for low crest factor.

        φₖ = -πk(k-1)/N
        """
        k = np.arange(N)
        return -np.pi * k * (k - 1) / N

    def estimate_kernel(
        self,
        excitation: ArrayF,
        response: ArrayF,
        frequencies: ArrayF,
        kernel_length: int = 512
    ) -> VolterraKernelFull:
        """
        Extract Volterra kernels from multi-tone measurement.

        Args:
            excitation: Input signal sent to system
            response: Output signal recorded from system
            frequencies: Tone frequencies used in excitation
            kernel_length: Length of extracted kernels

        Returns:
            VolterraKernelFull with diagonal kernels extracted
        """
        # FFT of signals
        X = fft.rfft(excitation)
        Y = fft.rfft(response)
        freqs_fft = fft.rfftfreq(len(excitation), 1/self.config.sample_rate)

        # Extract linear kernel (h1)
        h1 = self._extract_h1(X, Y, frequencies, freqs_fft, kernel_length)

        # Extract 2nd-order diagonal
        h2_diag = None
        if self.config.max_order >= 2:
            h2_diag = self._extract_h2_diagonal(X, Y, frequencies, freqs_fft, kernel_length)

        # Extract 3rd-order diagonal
        h3_diag = None
        if self.config.max_order >= 3:
            h3_diag = self._extract_h3_diagonal(X, Y, frequencies, freqs_fft, kernel_length)

        # Extract 4th-order diagonal
        h4_diag = None
        if self.config.max_order >= 4:
            h4_diag = self._extract_h4_diagonal(X, Y, frequencies, freqs_fft, kernel_length)

        # Extract 5th-order diagonal
        h5_diag = None
        if self.config.max_order >= 5:
            h5_diag = self._extract_h5_diagonal(X, Y, frequencies, freqs_fft, kernel_length)

        return VolterraKernelFull(
            h1=h1,
            h2=h2_diag,
            h3_diagonal=h3_diag,
            h4_diagonal=h4_diag,
            h5_diagonal=h5_diag,
            h2_is_diagonal=True
        )

    def _extract_h1(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        frequencies: ArrayF,
        freqs_fft: ArrayF,
        kernel_length: int
    ) -> ArrayF:
        """
        Extract linear kernel from fundamental frequencies.

        H1(f) = Y(f) / X(f)
        """
        H1_freq = np.zeros(len(freqs_fft), dtype=complex)

        for f in frequencies:
            bin_f = np.argmin(np.abs(freqs_fft - f))

            if np.abs(X[bin_f]) > 1e-12:
                H1_freq[bin_f] = Y[bin_f] / X[bin_f]

        # Inverse FFT to time domain
        h1_full = fft.irfft(H1_freq)
        return h1_full[:kernel_length]

    def _extract_h2_diagonal(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        frequencies: ArrayF,
        freqs_fft: ArrayF,
        kernel_length: int
    ) -> ArrayF:
        """
        Extract diagonal h2 from 2nd-harmonic content.

        Y(2f) ≈ (1/2) · H2_diag(f) · |X(f)|²
        """
        H2_diag_freq = np.zeros(len(freqs_fft), dtype=complex)

        for f in frequencies:
            bin_f = np.argmin(np.abs(freqs_fft - f))
            bin_2f = np.argmin(np.abs(freqs_fft - 2*f))

            if bin_2f < len(Y):
                X_f = X[bin_f]
                Y_2f = Y[bin_2f]

                if np.abs(X_f) > 1e-12:
                    # H2_diag(f) = 2 · Y(2f) / X(f)²
                    H2_diag_freq[bin_f] = 2 * Y_2f / (X_f**2)

        h2_diag_full = fft.irfft(H2_diag_freq)
        return h2_diag_full[:kernel_length]

    def _extract_h3_diagonal(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        frequencies: ArrayF,
        freqs_fft: ArrayF,
        kernel_length: int
    ) -> ArrayF:
        """
        Extract diagonal h3 from 3rd-harmonic content.

        Y(3f) ≈ (1/6) · H3_diag(f) · |X(f)|³
        """
        H3_diag_freq = np.zeros(len(freqs_fft), dtype=complex)

        for f in frequencies:
            bin_f = np.argmin(np.abs(freqs_fft - f))
            bin_3f = np.argmin(np.abs(freqs_fft - 3*f))

            if bin_3f < len(Y):
                X_f = X[bin_f]
                Y_3f = Y[bin_3f]

                if np.abs(X_f) > 1e-12:
                    # H3_diag(f) = 6 · Y(3f) / X(f)³
                    H3_diag_freq[bin_f] = 6 * Y_3f / (X_f**3)

        h3_diag_full = fft.irfft(H3_diag_freq)
        return h3_diag_full[:kernel_length]

    def _extract_h4_diagonal(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        frequencies: ArrayF,
        freqs_fft: ArrayF,
        kernel_length: int
    ) -> ArrayF:
        """
        Extract diagonal h4 from 4th-harmonic content.

        Y(4f) ≈ (1/24) · H4_diag(f) · |X(f)|⁴
        """
        H4_diag_freq = np.zeros(len(freqs_fft), dtype=complex)

        for f in frequencies:
            bin_f = np.argmin(np.abs(freqs_fft - f))
            bin_4f = np.argmin(np.abs(freqs_fft - 4*f))

            if bin_4f < len(Y):
                X_f = X[bin_f]
                Y_4f = Y[bin_4f]

                if np.abs(X_f) > 1e-12:
                    # H4_diag(f) = 24 · Y(4f) / X(f)⁴
                    H4_diag_freq[bin_f] = 24 * Y_4f / (X_f**4)

        h4_diag_full = fft.irfft(H4_diag_freq)
        return h4_diag_full[:kernel_length]

    def _extract_h5_diagonal(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        frequencies: ArrayF,
        freqs_fft: ArrayF,
        kernel_length: int
    ) -> ArrayF:
        """
        Extract diagonal h5 from 5th-harmonic content.

        Y(5f) ≈ (1/120) · H5_diag(f) · |X(f)|⁵
        """
        H5_diag_freq = np.zeros(len(freqs_fft), dtype=complex)

        for f in frequencies:
            bin_f = np.argmin(np.abs(freqs_fft - f))
            bin_5f = np.argmin(np.abs(freqs_fft - 5*f))

            if bin_5f < len(Y):
                X_f = X[bin_f]
                Y_5f = Y[bin_5f]

                if np.abs(X_f) > 1e-12:
                    # H5_diag(f) = 120 · Y(5f) / X(f)⁵
                    H5_diag_freq[bin_f] = 120 * Y_5f / (X_f**5)

        h5_diag_full = fft.irfft(H5_diag_freq)
        return h5_diag_full[:kernel_length]

    def estimate_snr_db(
        self,
        response: ArrayF,
        frequencies: ArrayF
    ) -> float:
        """
        Estimate SNR of measurement from noise floor.

        Returns:
            SNR in dB
        """
        Y = fft.rfft(response)
        freqs_fft = fft.rfftfreq(len(response), 1/self.config.sample_rate)

        # Find bins with signal (fundamentals + harmonics up to 5th)
        signal_bins = set()
        for f in frequencies:
            for k in range(1, 6):
                bin_kf = np.argmin(np.abs(freqs_fft - k*f))
                signal_bins.add(bin_kf)

        signal_bins = np.array(list(signal_bins))
        noise_bins = np.setdiff1d(np.arange(len(Y)), signal_bins)

        signal_power = np.mean(np.abs(Y[signal_bins])**2)
        noise_power = np.mean(np.abs(Y[noise_bins])**2)

        if noise_power > 0:
            return 10 * np.log10(signal_power / noise_power)
        else:
            return np.inf
