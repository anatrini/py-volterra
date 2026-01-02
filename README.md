# py-volterra

Production-ready **Tensor-Train based Volterra system identification** for nonlinear MIMO audio/acoustic systems.

Efficient diagonal Volterra (memory polynomial) identification using state-of-the-art Tensor-Train decomposition, avoiding the curse of dimensionality.

## Features

- **TT-Volterra Identification**: SISO and MIMO nonlinear system identification
- **Diagonal Volterra Models**: Memory polynomials / Generalized Hammerstein
- **Multiple Solvers**:
  - **ALS**: Batch fixed-rank (stationary systems, highest accuracy)
  - **RLS**: Online/adaptive (time-varying systems, real-time)
  - **MALS**: Adaptive-rank (experimental)
- **MIMO Support**: Additive model with separate kernels per input
- **Efficient**: O(M·T·N²) complexity, suitable for high-order (M=5+)
- **Numerically Stable**: Machine precision coefficient recovery
- **Production Ready**: 226 tests, 68% coverage, comprehensive documentation

## Installation

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
git clone https://github.com/anatrini/py-volterra.git
cd py-volterra
uv sync
```

Or with pip:

```bash
pip install numpy scipy
```

## Quick Start

### SISO System Identification

```python
import numpy as np
from volterra.models import TTVolterraIdentifier, TTVolterraConfig

# Generate training data
x_train = np.random.randn(5000) * 0.5
y_train = 0.8*x_train + 0.15*x_train**2 + 0.05*x_train**3  # Nonlinear system

# Identify model
identifier = TTVolterraIdentifier(
    memory_length=20,    # 20 samples of memory (~0.4ms @ 48kHz)
    order=3,             # Up to 3rd-order nonlinearity
    ranks=[1, 1, 1, 1],  # Diagonal (all ranks = 1)
    config=TTVolterraConfig(solver='als', max_iter=50)
)
identifier.fit(x_train, y_train)

# Predict on new data
x_test = np.random.randn(1000) * 0.5
y_pred = identifier.predict(x_test)

# Check accuracy
print(f"Final loss: {identifier.fit_info_['per_output'][0]['final_loss']:.6e}")
```

### MIMO System Identification

```python
# 2-channel input (e.g., stereo loudspeaker)
x_mimo = np.random.randn(5000, 2) * 0.5

# Each input has independent nonlinearity
y_mimo = (0.9*x_mimo[:, 0] + 0.15*x_mimo[:, 0]**2 +    # Left channel
          0.8*x_mimo[:, 1] + 0.10*x_mimo[:, 1]**2)     # Right channel

# Fit MIMO model
identifier_mimo = TTVolterraIdentifier(
    memory_length=15,
    order=2,
    ranks=[1, 1, 1],
    config=TTVolterraConfig(solver='als', max_iter=100)
)
identifier_mimo.fit(x_mimo, y_mimo)

# Predict
y_pred_mimo = identifier_mimo.predict(x_mimo[:1000])
```

### Online/Adaptive Filtering with RLS

```python
# Time-varying system
x = np.random.randn(10000)
y = np.zeros_like(x)

for t in range(len(x)):
    # Coefficient varies over time
    alpha = 0.8 + 0.2 * np.sin(2 * np.pi * t / 500)
    y[t] = alpha * x[t] + 0.1 * x[t]**2

# Adaptive identification
identifier_rls = TTVolterraIdentifier(
    memory_length=5,
    order=2,
    ranks=[1, 1, 1],
    config=TTVolterraConfig(
        solver='rls',           # Recursive Least Squares
        forgetting_factor=0.99,  # Track slow variations
        regularization=1e-4
    )
)
identifier_rls.fit(x, y)

# Track adaptation
import matplotlib.pyplot as plt
mse_history = identifier_rls.fit_info_['per_output'][0]['mse_history']
plt.semilogy(mse_history)
plt.xlabel('Sample')
plt.ylabel('MSE')
plt.title('RLS Adaptation')
plt.show()
```

## Mathematical Model

**Diagonal Volterra (Memory Polynomial):**

```
y(t) = ∑_{m=1}^M ∑_{i=0}^{N-1} h_m[i] · x[t-i]^m
```

- **M**: Volterra order (polynomial degree)
- **N**: Memory length (number of delays)
- **h_m[i]**: Kernel coefficient for order m at delay i

**Parameters:** M × N (vs O(N^M) for full Volterra)

**MIMO Additive Model:**

```
y(t) = ∑_{i=1}^I ∑_{m=1}^M ∑_{k=0}^{N-1} h_{i,m}[k] · x_i[t-k]^m
```

- Each input channel has its own set of Volterra kernels
- Outputs are summed

## Performance

| Configuration | Parameters | Complexity (per iter) | Use Case |
|--------------|------------|-----------------------|----------|
| M=3, N=10 | 30 | O(3·T·100) | Small systems |
| M=5, N=20 | 100 | O(5·T·400) | Audio effects |
| M=7, N=50 | 350 | O(7·T·2500) | High-fidelity |

**Convergence:** Typically 20-50 iterations

**Accuracy:** < 1e-20 MSE on clean data, machine precision coefficient recovery

## Solvers Comparison

| Solver | Type | Best For | MIMO | Time-varying |
|--------|------|----------|------|--------------|
| **als** | Batch | Stationary systems, highest accuracy | ✅ | ❌ |
| **rls** | Online | Time-varying, real-time | ⚠️ SISO only | ✅ |
| **mals** | Batch | Rank selection (experimental) | ⚠️ Placeholder | ❌ |

## Project Structure

```
py-volterra/
├── volterra/
│   ├── models/
│   │   ├── tt_volterra.py         # TTVolterraIdentifier (main API)
│   │   └── tt_predict.py          # Prediction functions
│   ├── tt/
│   │   ├── tt_solvers.py          # High-level solver interfaces
│   │   ├── tt_solvers_simple.py   # Diagonal TT-ALS and RLS
│   │   └── tt_tensor.py           # TT tensor representation
│   ├── pipelines/
│   │   └── acoustic_chain.py      # Nonlinear → RIR pipeline
│   └── utils/
│       └── shapes.py              # MIMO data utilities
├── tests/                         # 226 tests, 68% coverage
│   ├── test_tt_als_real.py       # Diagonal TT-ALS tests
│   ├── test_mimo_rls.py          # MIMO and RLS tests
│   └── test_tt_volterra_identifier.py
├── TT_ALS_IMPLEMENTATION.md      # Comprehensive documentation
└── README.md
```

## Documentation

- **[TT_ALS_IMPLEMENTATION.md](TT_ALS_IMPLEMENTATION.md)**: Complete technical guide
  - Mathematical formulation
  - Algorithm details
  - MIMO and RLS usage
  - Performance characteristics
  - 5 detailed examples

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test suites
uv run pytest tests/test_tt_als_real.py    # TT-ALS solver tests
uv run pytest tests/test_mimo_rls.py       # MIMO and RLS tests

# With coverage
uv run pytest --cov=volterra --cov-report=html
```

**Test Results:** 226 tests passing, 68% coverage

## Use Cases

### Audio/Acoustic Systems

- **Loudspeaker identification**: Nonlinear distortion modeling
- **Microphone calibration**: Capsule nonlinearity compensation
- **Amplifier emulation**: Tube/solid-state characteristic extraction
- **Room acoustics**: Nonlinear + RIR combined models
- **Echo cancellation**: Adaptive nonlinear acoustic echo

### Research & Development

- **Nonlinear system identification**: General Volterra model fitting
- **Time-varying systems**: Adaptive filtering with RLS
- **MIMO processing**: Multi-channel nonlinear effects
- **Benchmarking**: Compare against Novak synchronized swept-sine

## Comparison with Other Methods

| Method | This Implementation | Novak Swept-Sine | Multi-tone |
|--------|---------------------|------------------|------------|
| **Type** | Diagonal Volterra (TT-ALS) | Diagonal Volterra (freq.) | Diagonal Volterra (freq.) |
| **Domain** | Time | Frequency | Frequency |
| **MIMO** | ✅ Native | ⚠️ Per channel | ⚠️ Per channel |
| **Adaptive** | ✅ RLS | ❌ | ❌ |
| **Complexity** | O(M·T·N²) | O(T·log T) | O(T·log T) |
| **Memory** | M×N | M×N | M×N |

## Version History

**v0.7.0** (Current)
- ✅ MIMO support (additive diagonal Volterra)
- ✅ RLS online/adaptive solver
- ✅ 17 new tests (MIMO + RLS)
- ✅ Comprehensive documentation updates

**v0.6.1**
- ✅ Diagonal TT-ALS solver (memory polynomial)
- ✅ Sliding-window prediction
- ✅ 20 comprehensive tests
- ✅ Full documentation (TT_ALS_IMPLEMENTATION.md)

## References

1. Boyd, S., Tang, Y.Y., & Chua, L.O. (1983). "Measuring Volterra kernels", IEEE Trans. Circuits and Systems
2. Schetzen, M. (1980). "The Volterra and Wiener Theories of Nonlinear Systems", Wiley
3. Oseledets, I.V. (2011). "Tensor-Train Decomposition", SIAM J. Sci. Comput.
4. Novak, A. et al. (2015). "Synchronized Swept-Sine", JAES
5. Haykin, S. (2002). "Adaptive Filter Theory", Prentice Hall
6. Diniz, P.S.R. (2013). "Adaptive Filtering", Springer

## License

MIT

## Contributing

Issues and pull requests welcome at https://github.com/anatrini/py-volterra
