# py-volterra

Python implementation of Volterra series for nonlinear audio processing and harmonic distortion modeling (2nd/3rd order kernels).

## Features

- **2nd-order Volterra kernels** with enforced symmetry
- **Low-rank decomposition** for efficient computation (O(R·N) vs O(N²))
- **Block-based streaming** with proper state management
- **Farina swept-sine** method for kernel extraction
- Support for Hammerstein models (diagonal kernels)

## Quick Start with uv

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. No virtual environment activation needed!

### Installation

1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/anatrini/py-volterra.git
   cd py-volterra
   ```

3. **Install dependencies** (uv handles everything automatically):
   ```bash
   uv sync
   ```

### Running Examples

```bash
# Run validation tests
uv run python examples/validation.py

# Run performance benchmark
uv run python examples/benchmark.py

# Process audio (requires input.wav)
uv run python examples/demo.py
```

## Basic Usage

```python
import numpy as np
from volterra import VolterraKernel2, VolterraProcessor2, LowRankEngine

# Create a simple saturation kernel
N = 512
h1 = np.zeros(N); h1[0] = 0.9      # Linear path
g2 = np.zeros(N); g2[0] = 0.15     # 2nd-order saturation

kernel = VolterraKernel2.from_hammerstein(h1, g2)

# Process audio with low-rank optimization
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
├── volterra/           # Core package
│   ├── kernels.py      # Kernel data structures
│   ├── engines.py      # Processing engines (dense/low-rank)
│   ├── processor.py    # Streaming processor
│   └── sweep.py        # Farina swept-sine extraction
├── examples/           # Usage examples
│   ├── demo.py         # Audio processing demo
│   ├── validation.py   # Unit tests
│   └── benchmark.py    # Performance tests
└── pyproject.toml      # Project configuration
```

## Dependencies

- **numpy** ≥1.24: Core numerical operations
- **scipy** ≥1.10: Signal processing (lfilter, fftconvolve)
- **soundfile** (optional): Audio I/O for examples
- **matplotlib** (optional): Visualization

## Development

```bash
# Add new dependencies
uv add package-name

# Run scripts directly
uv run python your_script.py

# Run tests
uv run python examples/validation.py
```

## Technical Details

- **Kernel size**: N=512 samples (optimized for real-time processing)
- **Low-rank approximation**: Reduces complexity from O(N²) to O(R·N) where R≈20
- **Block processing**: Configurable block size (default 512 samples)
- **Harmonic extraction**: Farina's deconvolution method for system identification

## License

MIT

## References

- Farina, A. (2000). "Simultaneous measurement of impulse response and distortion with a swept-sine technique"
- Novák, A. et al. (2010). "Nonlinear System Identification Using Exponential Swept-Sine Signal"
