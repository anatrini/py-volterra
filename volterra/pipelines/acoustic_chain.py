"""
Acoustic chain: Nonlinear system → Room Impulse Response (RIR).

This module implements composable pipelines for realistic acoustic modeling:

    Source → [Nonlinearity] → [Room/Space] → Microphone

Where:
- Nonlinearity: Modeled by Volterra/TT system (instrument, PA, speaker)
- Room/Space: Modeled by linear convolution with RIR

This separation is physically motivated:
1. Source-side nonlinearities (guitar amp saturation, speaker cone movement)
2. Linear acoustic propagation (reflections, absorption, diffraction)

The NonlinearThenRIR class provides:
- Sequential composition of nonlinear → linear stages
- Efficient convolution-based RIR application
- Support for multi-channel RIR (MIMO)
- Compatible with TTVolterraIdentifier for nonlinear stage

Typical usage:
--------------
    from volterra.models import TTVolterraIdentifier
    from volterra.pipelines import NonlinearThenRIR
    import numpy as np
    from scipy.signal import fftconvolve

    # Stage 1: Identify nonlinear system (e.g., guitar amp)
    identifier = TTVolterraIdentifier(memory_length=20, order=3, ranks=[1, 5, 3, 1])
    identifier.fit(x_amp_input, y_amp_output)

    # Stage 2: Measure or load RIR
    rir = load_room_impulse_response()  # shape (RIR_length, n_channels)

    # Compose full chain
    chain = NonlinearThenRIR(
        nonlinear_model=identifier,
        rir=rir,
        sample_rate=48000
    )

    # Process input signal through full chain
    x_instrument = generate_instrument_signal()
    y_recorded = chain.process(x_instrument)  # Nonlinearity → Room → Output
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


@dataclass
class AcousticChainConfig:
    """
    Configuration for acoustic chain processing.

    Parameters
    ----------
    rir_method : str, default='fft'
        RIR convolution method: 'fft' (scipy.signal.fftconvolve) or 'direct' (np.convolve)
    normalize_rir : bool, default=False
        Normalize RIR to unit energy before convolution
    trim_output : bool, default=True
        Trim output to original signal length (remove RIR tail)
    """

    rir_method: str = "fft"
    normalize_rir: bool = False
    trim_output: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.rir_method not in ("fft", "direct"):
            raise ValueError(f"rir_method must be 'fft' or 'direct', got '{self.rir_method}'")


class NonlinearThenRIR:
    """
    Acoustic chain: Nonlinear Volterra system followed by Room Impulse Response.

    This class composes two stages:
    1. Nonlinear system (Volterra/TT model) - models source nonlinearity
    2. Linear convolution with RIR - models acoustic propagation

    The composition is:
        x(t) → [Nonlinear Model] → z(t) → [RIR Convolution] → y(t)

    Where:
    - x(t): Input signal (e.g., instrument, speech)
    - z(t): Nonlinear output (e.g., amplified signal)
    - y(t): Final output (e.g., recorded signal with room acoustics)

    Attributes
    ----------
    nonlinear_model : object
        Fitted nonlinear model with predict() method (e.g., TTVolterraIdentifier)
    rir : np.ndarray
        Room impulse response, shape (RIR_length,) for SISO or (RIR_length, n_channels) for MIMO
    sample_rate : int
        Sample rate in Hz
    config : AcousticChainConfig
        Processing configuration

    Examples
    --------
    >>> # SISO chain (mono → mono)
    >>> from volterra.models import TTVolterraIdentifier
    >>> from volterra.pipelines import NonlinearThenRIR
    >>> import numpy as np
    >>>
    >>> # Train nonlinear model
    >>> nl_model = TTVolterraIdentifier(memory_length=10, order=2, ranks=[1, 3, 1])
    >>> nl_model.fit(x_train, y_train)
    >>>
    >>> # Load/create RIR
    >>> rir = np.random.randn(4800) * np.exp(-np.arange(4800) / 4800)  # Simple decay
    >>>
    >>> # Create chain
    >>> chain = NonlinearThenRIR(nl_model, rir, sample_rate=48000)
    >>>
    >>> # Process signal
    >>> x = np.random.randn(10000)
    >>> y = chain.process(x)

    >>> # MIMO chain (mono → stereo)
    >>> rir_stereo = np.random.randn(4800, 2)  # Stereo RIR
    >>> chain_stereo = NonlinearThenRIR(nl_model, rir_stereo, sample_rate=48000)
    >>> y_stereo = chain_stereo.process(x)  # Output shape: (T, 2)
    """

    def __init__(
        self,
        nonlinear_model,
        rir: np.ndarray,
        sample_rate: int = 48000,
        config: AcousticChainConfig | None = None,
    ):
        """
        Initialize acoustic chain.

        Parameters
        ----------
        nonlinear_model : object
            Fitted nonlinear model with predict() method
        rir : np.ndarray
            Room impulse response, shape (RIR_length,) or (RIR_length, n_channels)
        sample_rate : int, default=48000
            Sample rate in Hz
        config : AcousticChainConfig, optional
            Processing configuration

        Raises
        ------
        ValueError
            If nonlinear_model is not fitted or RIR shape is invalid
        """
        # Validate nonlinear model
        if not hasattr(nonlinear_model, "predict"):
            raise ValueError("nonlinear_model must have predict() method")
        if hasattr(nonlinear_model, "is_fitted") and not nonlinear_model.is_fitted:
            raise ValueError("nonlinear_model must be fitted before use")

        # Validate and canonicalize RIR
        if rir.ndim == 1:
            rir = rir[:, np.newaxis]  # (L,) → (L, 1) for SISO
        elif rir.ndim != 2:
            raise ValueError(f"RIR must be 1D or 2D, got shape {rir.shape}")

        if rir.shape[0] < 1:
            raise ValueError("RIR cannot be empty")

        self.nonlinear_model = nonlinear_model
        self.rir = rir
        self.sample_rate = sample_rate
        self.config = config or AcousticChainConfig()

        # Process RIR if needed
        if self.config.normalize_rir:
            # Normalize each channel to unit energy
            for ch in range(self.rir.shape[1]):
                energy = np.sum(self.rir[:, ch] ** 2)
                if energy > 0:
                    self.rir[:, ch] /= np.sqrt(energy)

        self.rir_length = self.rir.shape[0]
        self.n_rir_channels = self.rir.shape[1]

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Process input signal through nonlinear → RIR chain.

        Parameters
        ----------
        x : np.ndarray
            Input signal, shape (T,) for SISO or (T, I) for MIMO

        Returns
        -------
        y : np.ndarray
            Output signal after nonlinearity and room acoustics.
            Shape (T,) for SISO or (T, O) for MIMO, where O = n_rir_channels

        Raises
        ------
        ValueError
            If input shape is incompatible

        Examples
        --------
        >>> y = chain.process(x)
        """
        # Stage 1: Nonlinear processing
        z = self.nonlinear_model.predict(x)  # Nonlinear output

        # Ensure z is 2D for consistent processing
        if z.ndim == 1:
            z = z[:, np.newaxis]  # (T,) → (T, 1)

        T_z, n_z_channels = z.shape

        # Stage 2: RIR convolution
        # For each output channel, convolve each z channel with corresponding RIR
        # Simplified: assume single z channel for now
        if n_z_channels > 1:
            # For multi-channel z, use first channel
            z_mono = z[:, 0]
        else:
            z_mono = z[:, 0]

        # Convolve with RIR (mono → multi-channel)
        y_list = []
        for ch in range(self.n_rir_channels):
            rir_ch = self.rir[:, ch]

            if self.config.rir_method == "fft":
                y_ch = fftconvolve(z_mono, rir_ch, mode="full")
            else:
                y_ch = np.convolve(z_mono, rir_ch, mode="full")

            # Trim to original length if requested
            if self.config.trim_output:
                y_ch = y_ch[:T_z]

            y_list.append(y_ch)

        # Stack channels
        y = np.column_stack(y_list)  # (T, n_rir_channels)

        # Return in original format
        if self.n_rir_channels == 1:
            return y[:, 0]  # (T,) for SISO
        else:
            return y  # (T, O) for MIMO

    def get_rir(self, channel: int = 0) -> np.ndarray:
        """
        Get RIR for specified channel.

        Parameters
        ----------
        channel : int, default=0
            Channel index (0-indexed)

        Returns
        -------
        rir_ch : np.ndarray
            RIR for channel, shape (RIR_length,)

        Raises
        ------
        ValueError
            If channel index out of range

        Examples
        --------
        >>> rir_left = chain.get_rir(channel=0)
        >>> rir_right = chain.get_rir(channel=1)
        """
        if channel < 0 or channel >= self.n_rir_channels:
            raise ValueError(f"channel must be in [0, {self.n_rir_channels-1}], got {channel}")
        return self.rir[:, channel]

    def get_nonlinear_model(self):
        """
        Get the nonlinear model (stage 1).

        Returns
        -------
        nonlinear_model : object
            The fitted nonlinear model

        Examples
        --------
        >>> nl_model = chain.get_nonlinear_model()
        >>> kernels = nl_model.get_kernels()  # If TTVolterraIdentifier
        """
        return self.nonlinear_model

    @property
    def is_mimo(self) -> bool:
        """Whether output is multi-channel (MIMO)."""
        return self.n_rir_channels > 1

    def __repr__(self) -> str:
        """String representation."""
        nl_name = type(self.nonlinear_model).__name__
        return (
            f"NonlinearThenRIR("
            f"nonlinear={nl_name}, "
            f"rir_length={self.rir_length}, "
            f"rir_channels={self.n_rir_channels}, "
            f"fs={self.sample_rate}Hz)"
        )
