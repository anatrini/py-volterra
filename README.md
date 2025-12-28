# py-volterra

Python implementation of Volterra series for nonlinear audio processing and harmonic distortion modeling (up to 5th order kernels).

## Features

- **Volterra kernels up to 5th order** for high-fidelity analog emulation
- **Diagonal approximation** for real-time performance (O(N) memory vs O(N^k))
- **2nd-order with full matrix support** and low-rank decomposition
- **Multi-tone kernel estimation** for system identification
- **Optimized engines**: NumPy (portable) and Numba (10x faster)
- **Farina swept-sine** method for 2nd-order extraction
- **Block-based streaming** with proper state management

## Quick Start with uv

[uv](https://github.com/astral-sh/uv) is a fast Python package manager that handles everything automatically.

### Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup**:
   ```bash
   git clone https://github.com/anatrini/py-volterra.git
   cd py-volterra
   uv sync
   ```

3. **Optional: Install performance optimization (Numba)**:
   ```bash
   uv sync --extra performance
   ```

### Running Examples

```bash
# Full-order demo (orders 1-5)
uv run python examples/demo_full_order.py

# Validation tests
uv run python examples/validation_full_order.py

# Performance benchmark
uv run python examples/benchmark_full_order.py

# Legacy 2nd-order demo
uv run python examples/demo.py
```

## Basic Usage

### Simple Polynomial Saturation (Orders 1-5)

```python
from volterra import VolterraKernelFull, VolterraProcessorFull

# Create a tube-style saturation kernel
# y ≈ 0.9x + 0.12x² + 0.03x³ + 0.005x⁴ + 0.008x⁵
kernel = VolterraKernelFull.from_polynomial_coeffs(
    N=512,
    a1=0.9,    # Linear gain
    a2=0.12,   # Even harmonics (warmth)
    a3=0.03,   # Odd harmonics (character)
    a4=0.005,  # Even harmonics (subtlety)
    a5=0.008   # Odd harmonics (edge)
)

# Process audio with automatic Numba optimization
proc = VolterraProcessorFull(kernel, sample_rate=48000)
output = proc.process(input_audio, block_size=512)
```

### Multi-tone Kernel Estimation

```python
from volterra import MultiToneConfig, MultiToneEstimator

# Configure multi-tone measurement
config = MultiToneConfig(
    sample_rate=48000,
    duration=2.0,
    num_tones=100,
    max_order=5  # Extract kernels up to 5th order
)

# Generate excitation signal
estimator = MultiToneEstimator(config)
excitation, frequencies = estimator.generate_excitation()

# Send excitation through your system, record response
# (e.g., through analog gear, tube amp, etc.)
response = your_system.process(excitation)

# Extract kernels
kernel = estimator.estimate_kernel(excitation, response, frequencies)

# Use extracted kernel
proc = VolterraProcessorFull(kernel, sample_rate=48000)
```

### Legacy 2nd-Order API (Backward Compatible)

```python
from volterra import VolterraKernel2, VolterraProcessor2, LowRankEngine

# 2nd-order with full matrix and low-rank approximation
N = 512
h1 = np.zeros(N); h1[0] = 0.9
g2 = np.zeros(N); g2[0] = 0.15

kernel = VolterraKernel2.from_hammerstein(h1, g2)

proc = VolterraProcessor2(
    kernel=kernel,
    engine=LowRankEngine(energy=0.999),
    sample_rate=48000
)

y = proc.process(x, block_size=512)
```

## Project Structure

```
py-volterra/
├── volterra/                      # Core package
│   ├── kernels.py                 # 2nd-order kernels (legacy)
│   ├── kernels_full.py            # Full kernels (orders 1-5)
│   ├── engines.py                 # 2nd-order engines (legacy)
│   ├── engines_diagonal.py        # Diagonal engines (orders 1-5)
│   ├── processor.py               # 2nd-order processor (legacy)
│   ├── processor_full.py          # Full-order processor
│   ├── estimation.py              # Multi-tone kernel estimation
│   └── sweep.py                   # Farina swept-sine extraction
├── examples/
│   ├── demo_full_order.py         # Full demo (orders 1-5)
│   ├── validation_full_order.py   # Validation tests
│   ├── benchmark_full_order.py    # Performance benchmark
│   ├── demo.py                    # Legacy 2nd-order demo
│   ├── validation.py              # Legacy validation
│   └── benchmark.py               # Legacy benchmark
└── pyproject.toml
```

## Performance

### Memory Footprint (N=512)

| Configuration | Memory | Notes |
|---------------|--------|-------|
| h1 only | 4 KB | Linear convolution |
| h1 + h2 (diagonal) | 8 KB | 2nd-order diagonal |
| h1 + h2 (full) | 2 MB | 2nd-order full matrix |
| h1 + h2 + h3 | 12 KB | Up to 3rd order |
| h1 + h2 + h3 + h5 | 16 KB | Up to 5th order |
| All orders (1-5) | 20 KB | Complete saturation model |

### Processing Latency (N=512, block_size=512)

With Numba (recommended):
- **h1 only**: ~0.1 ms/block
- **h1 + h2**: ~0.3 ms/block
- **h1 + h2 + h3**: ~0.5 ms/block
- **h1 + h2 + h3 + h5**: ~0.8 ms/block (✓ real-time @ 48kHz)
- **All orders (1-5)**: ~1.0 ms/block (✓ real-time @ 48kHz)

Without Numba (NumPy only):
- ~10x slower (still real-time for lower orders)

## Dependencies

**Required:**
- numpy ≥1.24
- scipy ≥1.10

**Optional (recommended for performance):**
- numba ≥0.58 (10x speedup for orders 3-5)

**Optional (for examples):**
- soundfile ≥0.12 (audio I/O)
- matplotlib ≥3.7 (visualization)

Install with extras:
```bash
uv sync --extra performance    # Add Numba
uv sync --extra all            # Everything
uv sync --extra dev            # Development
```

## Technical Details

### Diagonal Approximation

For orders k ≥ 3, storing full kernels is impractical:
- Full h3: 512³ × 8 bytes = 1 GB
- Diagonal h3: 512 × 8 bytes = 4 KB (99.6% reduction)

The diagonal approximation:
```
y_k[n] = Σᵢ hk_diag[i] · x[n-i]^k
```

captures 70-80% of nonlinear energy for typical audio distortion while enabling real-time processing.

### Multi-tone Estimation Theory

Frequency-domain relationships for diagonal kernels:
- **Linear**: Y(f) = H1(f)·X(f)
- **2nd-order**: Y(2f) = ½·H2(f)·X(f)²
- **3rd-order**: Y(3f) = ⅙·H3(f)·X(f)³
- **4th-order**: Y(4f) = 1/24·H4(f)·X(f)⁴
- **5th-order**: Y(5f) = 1/120·H5(f)·X(f)⁵

By analyzing harmonic content, kernels are separated in frequency domain and inverted to time domain.

### Optimization Strategy

1. **Diagonal kernels** reduce memory from O(N^k) to O(N)
2. **Numba JIT compilation** provides 10x speedup via SIMD and parallelization
3. **Block processing** enables streaming with minimal latency
4. **Efficient convolution** using optimized inner loops

## Mathematical Accuracy

All implementations are validated against:
- Known polynomial systems (error < 0.1%)
- Harmonic separation tests
- Energy conservation
- Multi-tone estimation accuracy
- Engine consistency (Numba vs NumPy)

Run validation suite:
```bash
uv run python examples/validation_full_order.py
```

## Development

```bash
# Install with development tools
uv sync --extra dev

# Run all tests
uv run python examples/validation_full_order.py

# Benchmark performance
uv run python examples/benchmark_full_order.py

# Add dependencies
uv add package-name
```

## Use Cases

- **Analog gear emulation**: Tube amplifiers, tape saturation
- **Harmonic enhancement**: Musical distortion, warmth
- **System identification**: Measure nonlinear audio systems
- **Research**: Nonlinear audio processing, Volterra theory
- **Plugin development**: Real-time audio effects

## License

MIT

## References

- Farina, A. (2000). "Simultaneous measurement of impulse response and distortion with a swept-sine technique"
- Novák, A. et al. (2015). "Nonlinear System Identification Using Exponential Swept-Sine Signal"
- Reed & Hawksford (1996). "Identification of discrete Volterra series"
- Schetzen, M. (1980). "The Volterra and Wiener Theories of Nonlinear Systems"
