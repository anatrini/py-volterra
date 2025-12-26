import soundfile as sf
import numpy as np
from volterra import VolterraKernel2, VolterraProcessor2, LowRankEngine

# Load audio
x, fs = sf.read("input.wav", dtype="float64", always_2d=False)
assert fs == 48000, "Resample to 48kHz"

# Create mild saturation kernel (diagonal h2 for Phase 1)
N = 512
h1 = np.zeros(N); h1[0] = 0.9  # Slight gain reduction
g2 = np.zeros(N); g2[0] = 0.15  # Memoryless even-harmonic distortion

kernel = VolterraKernel2.from_hammerstein(h1, g2)

# Process with low-rank engine (the only way N=512 is fast enough)
proc = VolterraProcessor2(
    kernel=kernel,
    engine=LowRankEngine(energy=0.999),  # R≈1 for diagonal h2
    sample_rate=fs
)

y = proc.process(x, block_size=512)
sf.write("output.wav", y, fs)

# Save kernel for reuse
kernel.to_npz("saturation_kernel.npz")