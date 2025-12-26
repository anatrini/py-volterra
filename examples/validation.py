import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq


def test_delta_kernel():
    """Test 1: h1=δ, h2=0 should be identity."""
    from volterra import VolterraKernel2, VolterraProcessor2
    
    N = 512
    h1 = np.zeros(N); h1[0] = 1.0
    h2 = np.zeros((N, N))
    kernel = VolterraKernel2(h1, h2)
    
    proc = VolterraProcessor2(kernel)
    x = np.random.randn(10000)
    y = proc.process(x)
    
    error = np.max(np.abs(y - x))
    assert error < 1e-10, f"Delta test failed: error={error}"
    print("✓ Delta kernel test passed")


def test_even_harmonics():
    """Test 2: Pure quadratic should produce 2f, 4f, 6f..."""
    from volterra import VolterraKernel2, VolterraProcessor2
    
    N = 512
    h1 = np.zeros(N)  # No linear term
    h2 = np.zeros((N, N))
    h2[0, 0] = 0.3  # Pure memoryless quadratic
    kernel = VolterraKernel2(h1, h2)
    
    fs = 48000
    f0 = 1000.0
    t = np.arange(2 * fs) / fs
    x = np.sin(2 * np.pi * f0 * t)
    
    proc = VolterraProcessor2(kernel, sample_rate=fs)
    y = proc.process(x)
    
    # FFT analysis
    Y = rfft(y)
    freqs = rfftfreq(len(y), 1/fs)
    mag_db = 20 * np.log10(np.abs(Y) + 1e-12)
    
    def harmonic_level(k: int) -> float:
        idx = np.argmin(np.abs(freqs - k * f0))
        return mag_db[idx]
    
    h1_level = harmonic_level(1)  # Fundamental (should be low)
    h2_level = harmonic_level(2)  # 2nd harmonic (should be strong)
    h3_level = harmonic_level(3)  # 3rd harmonic (should be low)
    
    print(f"Harmonic levels: H1={h1_level:.1f} dB, H2={h2_level:.1f} dB, H3={h3_level:.1f} dB")
    assert h2_level - h1_level > 40, "Even harmonics not dominant"
    print("✓ Even harmonic separation verified")


def measure_thd(x: np.ndarray, y: np.ndarray, fs: int, f0: float) -> dict:
    """Calculate Total Harmonic Distortion."""
    Y = rfft(y)
    freqs = rfftfreq(len(y), 1/fs)
    
    fund_idx = np.argmin(np.abs(freqs - f0))
    fund_power = np.abs(Y[fund_idx]) ** 2
    
    harmonic_powers = []
    for k in range(2, 7):
        h_idx = np.argmin(np.abs(freqs - k * f0))
        harmonic_powers.append(np.abs(Y[h_idx]) ** 2)
    
    thd = np.sqrt(np.sum(harmonic_powers) / fund_power)
    return {
        'thd_percent': thd * 100,
        'thd_db': 20 * np.log10(thd),
        'harmonic_powers': harmonic_powers
    }


if __name__ == "__main__":
    test_delta_kernel()
    test_even_harmonics()