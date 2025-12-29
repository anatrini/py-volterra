"""
Complete demonstration of Volterra processing up to 5th order.

This example shows:
1. Creating kernels for analog-style saturation (orders 1-5)
2. Processing audio with the full-order processor
3. Analyzing harmonic content
4. Comparing different order combinations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not available. Install with: pip install soundfile")

from volterra import (
    VolterraKernelFull,
    VolterraProcessorFull,
)


def create_tube_saturation_kernel(N: int = 512) -> VolterraKernelFull:
    """
    Create a realistic tube-saturation kernel.

    Characteristics:
    - Slight gain reduction (linear)
    - Even harmonics dominant (h2, h4)
    - Odd harmonics for asymmetric clipping (h3, h5)
    - Memory effects through kernel length

    This approximates a triode tube characteristic:
    y ≈ 0.9x + 0.12x² + 0.03x³ + 0.005x⁴ + 0.008x⁵
    """
    # Linear kernel with slight lowpass character
    h1 = np.zeros(N, dtype=np.float64)
    h1[0] = 0.9
    h1[1:10] = 0.05 * np.exp(-np.arange(1, 10) / 3.0)  # Slight memory

    # 2nd order (even harmonics) - dominant
    h2_diag = np.zeros(N, dtype=np.float64)
    h2_diag[0] = 0.12
    h2_diag[1:5] = 0.01 * np.exp(-np.arange(1, 5) / 2.0)

    # 3rd order (odd harmonics) - moderate
    h3_diag = np.zeros(N, dtype=np.float64)
    h3_diag[0] = 0.03
    h3_diag[1:5] = 0.005 * np.exp(-np.arange(1, 5) / 2.0)

    # 4th order (even harmonics) - subtle
    h4_diag = np.zeros(N, dtype=np.float64)
    h4_diag[0] = 0.005

    # 5th order (odd harmonics) - adds "character"
    h5_diag = np.zeros(N, dtype=np.float64)
    h5_diag[0] = 0.008

    return VolterraKernelFull(
        h1=h1,
        h2=h2_diag,
        h3_diagonal=h3_diag,
        h4_diagonal=h4_diag,
        h5_diagonal=h5_diag,
        h2_is_diagonal=True
    )


def analyze_harmonics(signal: np.ndarray, fs: int, f0: float, num_harmonics: int = 10):
    """
    Analyze harmonic content of a signal.

    Returns:
        Dictionary with harmonic levels in dB
    """
    Y = rfft(signal)
    freqs = rfftfreq(len(signal), 1/fs)

    harmonics = {}
    for k in range(1, num_harmonics + 1):
        idx = np.argmin(np.abs(freqs - k * f0))
        level_db = 20 * np.log10(np.abs(Y[idx]) + 1e-12)
        harmonics[f"H{k}"] = level_db

    return harmonics


def demo_audio_processing():
    """Demonstrate full-order Volterra processing on audio."""
    print("="*70)
    print("VOLTERRA FULL-ORDER AUDIO PROCESSING DEMO")
    print("="*70)

    # Create kernel
    print("\n1. Creating tube saturation kernel (orders 1-5)...")
    kernel = create_tube_saturation_kernel(N=512)

    print(f"   Kernel length: {kernel.N}")
    print(f"   Max order: {kernel.max_order}")
    print(f"   Memory: {kernel.estimate_memory_bytes()/1024:.1f} KB")

    # Create processor
    print("\n2. Initializing processor...")
    proc = VolterraProcessorFull(kernel, sample_rate=48000, use_numba=True)
    print(f"   Engine: {proc.engine.__class__.__name__}")

    # Generate test signal (sine wave)
    fs = 48000
    duration = 1.0
    f0 = 1000.0  # 1 kHz test tone

    t = np.arange(int(fs * duration)) / fs
    x = 0.3 * np.sin(2 * np.pi * f0 * t)

    print(f"\n3. Processing test signal...")
    print(f"   Signal: {f0:.0f} Hz sine, {duration:.1f}s, amplitude=0.3")

    # Process
    y = proc.process(x, block_size=512)

    print(f"   Output RMS: {np.sqrt(np.mean(y**2)):.4f}")
    print(f"   Output peak: {np.max(np.abs(y)):.4f}")

    # Analyze harmonics
    print(f"\n4. Harmonic analysis:")
    harmonics_in = analyze_harmonics(x, fs, f0)
    harmonics_out = analyze_harmonics(y, fs, f0)

    print(f"   {'Harmonic':<10} {'Input (dB)':<12} {'Output (dB)':<12} {'Added (dB)'}")
    print(f"   {'-'*50}")

    for k in range(1, 7):
        h_in = harmonics_in.get(f"H{k}", -120)
        h_out = harmonics_out.get(f"H{k}", -120)
        added = h_out - h_in if h_in > -100 else h_out

        print(f"   {f'H{k} ({k*f0:.0f}Hz)':<10} {h_in:>10.1f}   {h_out:>10.1f}   {added:>10.1f}")

    # Calculate THD
    fund_power = 10**(harmonics_out["H1"] / 10)
    harmonic_power = sum(10**(harmonics_out[f"H{k}"] / 10) for k in range(2, 7))
    thd_percent = 100 * np.sqrt(harmonic_power / fund_power)

    print(f"\n   Total Harmonic Distortion: {thd_percent:.2f}%")

    # Save if soundfile available
    if SOUNDFILE_AVAILABLE:
        try:
            sf.write("demo_input.wav", x, fs)
            sf.write("demo_output_full_order.wav", y, fs)
            print(f"\n5. Audio files saved:")
            print(f"   - demo_input.wav")
            print(f"   - demo_output_full_order.wav")
        except Exception as e:
            print(f"\n   Warning: Could not save audio files: {e}")

    # Plot results
    plot_results(x, y, fs, f0)

    return x, y, kernel, proc


def plot_results(x, y, fs, f0):
    """Visualize time-domain and frequency-domain results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time domain
    t_plot = np.arange(min(1000, len(x))) / fs * 1000  # First 1000 samples in ms
    axes[0, 0].plot(t_plot, x[:len(t_plot)], label='Input', alpha=0.7)
    axes[0, 0].plot(t_plot, y[:len(t_plot)], label='Output (5th order)', alpha=0.7)
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Time Domain')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Frequency spectrum
    X = rfft(x)
    Y = rfft(y)
    freqs = rfftfreq(len(x), 1/fs)

    axes[0, 1].plot(freqs, 20*np.log10(np.abs(X) + 1e-12), label='Input', alpha=0.7)
    axes[0, 1].plot(freqs, 20*np.log10(np.abs(Y) + 1e-12), label='Output', alpha=0.7)
    axes[0, 1].set_xlim([0, 10000])
    axes[0, 1].set_ylim([-120, 0])
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].set_title('Frequency Spectrum')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Harmonic bars
    harmonics_out = analyze_harmonics(y, fs, f0, num_harmonics=10)
    harm_nums = list(range(1, 11))
    harm_levels = [harmonics_out[f"H{k}"] for k in harm_nums]

    axes[1, 0].bar(harm_nums, harm_levels, color='steelblue', alpha=0.7)
    axes[1, 0].set_xlabel('Harmonic Number')
    axes[1, 0].set_ylabel('Level (dB)')
    axes[1, 0].set_title('Harmonic Spectrum')
    axes[1, 0].grid(True, axis='y')

    # Transfer characteristic (input vs output)
    x_sweep = np.linspace(-0.5, 0.5, 1000)

    # Create small kernel for instant evaluation
    kernel_instant = VolterraKernelFull.from_polynomial_coeffs(
        N=1, a1=0.9, a2=0.12, a3=0.03, a4=0.005, a5=0.008
    )
    proc_instant = VolterraProcessorFull(kernel_instant, use_numba=False)
    y_sweep = np.array([proc_instant.process_block(np.array([xi]))[0] for xi in x_sweep])

    axes[1, 1].plot(x_sweep, x_sweep, 'k--', label='Linear', alpha=0.3)
    axes[1, 1].plot(x_sweep, y_sweep, 'r-', label='Volterra 5th order', linewidth=2)
    axes[1, 1].set_xlabel('Input')
    axes[1, 1].set_ylabel('Output')
    axes[1, 1].set_title('Transfer Characteristic')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('demo_full_order_results.png', dpi=150)
    print("\n   Plot saved: demo_full_order_results.png")


def demo_order_comparison():
    """Compare different Volterra orders."""
    print("\n" + "="*70)
    print("ORDER COMPARISON")
    print("="*70)

    fs = 48000
    f0 = 1000.0
    t = np.arange(fs) / fs
    x = 0.3 * np.sin(2 * np.pi * f0 * t)

    # Test different order combinations
    configs = [
        ("Linear only", dict(a1=0.9, a2=0, a3=0, a4=0, a5=0)),
        ("Up to 2nd", dict(a1=0.9, a2=0.12, a3=0, a4=0, a5=0)),
        ("Up to 3rd", dict(a1=0.9, a2=0.12, a3=0.03, a4=0, a5=0)),
        ("Up to 5th", dict(a1=0.9, a2=0.12, a3=0.03, a4=0.005, a5=0.008)),
    ]

    print(f"\nProcessing {f0:.0f} Hz sine wave with different orders:\n")
    print(f"{'Configuration':<15} {'THD (%)':<10} {'H2 (dB)':<10} {'H3 (dB)':<10} {'H5 (dB)'}")
    print("-"*70)

    for name, coeffs in configs:
        kernel = VolterraKernelFull.from_polynomial_coeffs(N=512, **coeffs)
        proc = VolterraProcessorFull(kernel, sample_rate=fs, use_numba=False)
        y = proc.process(x, block_size=512)

        harmonics = analyze_harmonics(y, fs, f0)

        # Calculate THD
        fund_power = 10**(harmonics["H1"] / 10)
        harmonic_power = sum(10**(harmonics[f"H{k}"] / 10) for k in range(2, 7))
        thd = 100 * np.sqrt(harmonic_power / fund_power)

        print(f"{name:<15} {thd:>8.2f}   {harmonics['H2']:>8.1f}   "
              f"{harmonics['H3']:>8.1f}   {harmonics['H5']:>8.1f}")


if __name__ == "__main__":
    # Run demos
    x, y, kernel, proc = demo_audio_processing()
    demo_order_comparison()

    print("\n" + "="*70)
    print("Demo complete! Check the generated files:")
    print("  - demo_full_order_results.png")
    if SOUNDFILE_AVAILABLE:
        print("  - demo_input.wav")
        print("  - demo_output_full_order.wav")
    print("="*70)
