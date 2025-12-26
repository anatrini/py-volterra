"""
CRITICAL COUNCIL FINDING:
Farina's method gives you harmonic impulse responses (diagonal/Hammerstein terms),
NOT a full 2D h2[i,j] from a single sweep.

For full off-diagonal h2: requires multi-tone grids or regularized fitting.
For Phase 1: diagonal h2 is a solid baseline for tube/tape saturation.
"""

import numpy as np
from scipy.signal import fftconvolve
from volterra.kernels import ArrayF


def exponential_sweep(fs: int, T: float, f0: float = 20.0, 
                     f1: float = 20000.0) -> ArrayF:
    """Generate Farina exponential sweep."""
    t = np.arange(int(fs * T), dtype=np.float64) / fs
    L = T / np.log(f1 / f0)
    K = T * 2 * np.pi * f0 / np.log(f1 / f0)
    phase = K * (np.exp(t / L) - 1.0)
    return np.sin(phase)


def inverse_filter(sweep: ArrayF, fs: int, T: float, 
                  f0: float, f1: float) -> ArrayF:
    """Farina inverse filter: time-reversed with amplitude correction."""
    t = np.arange(len(sweep), dtype=np.float64) / fs
    L = T / np.log(f1 / f0)
    env = np.exp(-t / L)
    return sweep[::-1] * env


def extract_harmonic_irs(y_recorded: ArrayF, sweep: ArrayF, 
                         fs: int, T: float, f0: float, f1: float,
                         N: int = 512) -> tuple[ArrayF, ArrayF]:
    """
    Extract h1 and diagonal g2 from Farina deconvolution.
    
    Returns:
        h1_est: linear IR (N,)
        g2_est: 2nd-order diagonal term (N,) for y₂ = g₂ ⊛ x²
    """
    inv = inverse_filter(sweep, fs, T, f0, f1)
    ir_full = fftconvolve(y_recorded, inv, mode='full')
    
    # Harmonic time shifts (Farina's key insight)
    dt2 = T * np.log(2) / np.log(f1 / f0)  # 2nd harmonic offset
    shift2_samples = int(round(dt2 * fs))
    
    # Find anchor (linear IR peak)
    anchor = int(np.argmax(np.abs(ir_full)))
    
    # Extract with Hann window
    w = np.hanning(N)
    
    def extract(center: int) -> ArrayF:
        start = max(center - N//2, 0)
        seg = ir_full[start:start+N]
        if len(seg) < N:
            seg = np.pad(seg, (0, N - len(seg)))
        return seg * w
    
    h1 = extract(anchor)
    g2 = extract(anchor - shift2_samples)
    
    return h1, g2