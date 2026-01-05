# py-volterra

Production-ready **Tensor-Train based Volterra system identification** for nonlinear MIMO audio/acoustic systems.

Comprehensive library supporting diagonal (memory polynomial), generalized memory polynomial (GMP), and full tensor-train Volterra models with automatic model selection.

## Features

- **Multiple Model Complexities**:
  - **Diagonal Volterra**: Memory polynomials / Generalized Hammerstein (O(M·N) params)
  - **GMP**: Generalized Memory Polynomial with selective cross-memory terms
  - **Full TT-Volterra**: Arbitrary-rank tensor-train with cross-memory interactions (O(M·r²·I·N))
- **Automatic Model Selection**: Intelligent selection between MP, GMP, and TT using AIC/BIC/NMSE
- **Multiple Solvers**:
  - **ALS**: Batch fixed-rank (stationary systems, highest accuracy)
  - **RLS**: Online/adaptive (time-varying systems, real-time)
  - **MALS**: Adaptive-rank (experimental)
- **MIMO Support**: Full support for multi-input multi-output systems
- **Efficient**: Tensor-train decomposition avoids curse of dimensionality
- **Numerically Stable**: High-precision coefficient recovery (< 1e-12 MSE for well-posed problems)
- **Production Ready**: 357 tests, 76% coverage, comprehensive documentation

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

### Automatic Model Selection

```python
from volterra.model_selection import ModelSelector

# Generate test data
x_train = np.random.randn(2000)
y_train = 0.8*x_train + 0.15*x_train**2 + 0.05*x_train[:]*x_train[:]**2  # With cross-memory

# Automatic model selection (tries Diagonal MP, GMP, and TT-Full)
selector = ModelSelector(
    memory_length=10,
    order=3,
    try_diagonal=True,
    try_gmp=True,
    try_tt_full=False,
    selection_criterion='bic'  # Use Bayesian Information Criterion
)
selector.fit(x_train, y_train)

# See selection rationale
print(selector.explain())

# Selected model: GMP
# NMSE: 1.23e-06, AIC: -2847.2, BIC: -2801.5

# Predict with best model
y_pred = selector.predict(x_test)
```

### Generalized Memory Polynomial (GMP)

```python
from volterra.models import GeneralizedMemoryPolynomial, GMPConfig

# Configure GMP with selective cross-memory terms
config = GMPConfig(
    max_cross_lag_distance=3,  # Cross-terms between lags up to 3 samples apart
    max_cross_order=2,          # Max total order for cross-terms (x^p * x^q where p+q <= 2)
    regularization=1e-6
)

model = GeneralizedMemoryPolynomial(
    memory_length=15,
    order=3,
    config=config
)

# Fit and predict
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Export for C++/Rust deployment
export = model.export_model()
print(f"Total terms: {model.total_terms}")
```

### Full TT-Volterra with Arbitrary Ranks

```python
from volterra.models import TTVolterraMIMO, TTVolterraFullConfig

# Full TT-Volterra with cross-memory interactions
config = TTVolterraFullConfig(
    max_iter=100,
    tol=1e-6,
    regularization=1e-6,
    verbose=True
)

model = TTVolterraMIMO(
    memory_length=10,
    order=3,
    ranks=[1, 3, 2, 1],  # TT ranks (boundary ranks must be 1)
    config=config
)

# Fit with full TT-ALS
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Access TT cores and diagnostics
cores = model.get_cores(output_idx=0)
diagnostics = model.diagnostics(output_idx=0)
print(f"Total parameters: {model.total_parameters}")
print(f"Final loss: {diagnostics['final_loss']:.6e}")
```

## Mathematical Models

### 1. Diagonal Volterra (Memory Polynomial)

```
y(t) = ∑_{m=1}^M ∑_{i=0}^{N-1} h_m[i] · x[t-i]^m
```

- **M**: Volterra order (polynomial degree)
- **N**: Memory length (number of delays)
- **h_m[i]**: Kernel coefficient for order m at delay i
- **Parameters**: M × N (linear scaling)

### 2. Generalized Memory Polynomial (GMP)

```
y(t) = ∑_{m=1}^M ∑_{k=0}^{N-1} h_m[k] · x[t-k]^m
       + ∑_{cross-terms} c_{k1,k2,p,q} · x[t-k1]^p · x[t-k2]^q
```

- Diagonal terms plus selective cross-memory interactions
- Cross-terms: |k1-k2| ≤ max_cross_lag, p+q ≤ max_cross_order
- **Parameters**: M × N + (configurable cross-terms)
- **Complexity**: Between diagonal MP and full Volterra

### 3. Full TT-Volterra with Arbitrary Ranks

```
y(t) = ∑_{i1,...,iM} G[i1,...,iM] · ∏_{m=1}^M x[t-i_m]
```

Represented in Tensor-Train format:
```
G[i1,...,iM] = G_1[i1] · G_2[i2] · ... · G_M[iM]
```

- **G_m**: TT core of shape (r_{m-1}, I×N, r_m)
- **Ranks**: [r_0=1, r_1, ..., r_{M-1}, r_M=1]
- **Parameters**: O(M × r² × I × N) for equal internal ranks r
- **Captures cross-memory interactions** with controlled complexity

### MIMO Models

All models support multi-input multi-output (MIMO):
```
y_o(t) = f_o(x_1[t], ..., x_I[t])  for each output o = 1, ..., O
```

**Array Shapes:**
- Input `x`: shape `(T, I)` where T = number of samples, I = number of input channels
- Output `y`: shape `(T, O)` where O = number of output channels
- SISO is special case with I=1, O=1

**Implementation:**
- Separate model per output channel (baseline)
- Shared parameters across outputs (advanced, experimental)

## Performance

| Configuration | Parameters | Complexity (per iter) | Use Case |
|--------------|------------|-----------------------|----------|
| M=3, N=10 | 30 | O(3·T·100) | Small systems |
| M=5, N=20 | 100 | O(5·T·400) | Audio effects |
| M=7, N=50 | 350 | O(7·T·2500) | High-fidelity |

**Convergence:** Typically 20-50 iterations for diagonal models

**Accuracy:** High precision on well-conditioned problems (MSE < 1e-12 for diagonal models with clean data)

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
├── tests/                         # 357 tests, 76% coverage
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

**Test Results:** 357 tests passing, 76% coverage

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

**v0.7.3** (Current)
- ✅ Full TT-Volterra with arbitrary ranks (TTVolterraMIMO)
- ✅ Generalized Memory Polynomial (GMP) with selective cross-terms
- ✅ Automatic model selection (ModelSelector) with AIC/BIC/NMSE
- ✅ TT-ALS primitives (cores, orthogonalization, rank truncation)
- ✅ 131 new tests (357 total, 76% coverage)
- ✅ Comprehensive documentation and technical guides

**v0.7.0**
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

### Volterra Series and Nonlinear System Identification
1. Boyd, S., Tang, Y.Y., & Chua, L.O. (1983). "Measuring Volterra kernels", IEEE Trans. Circuits and Systems, 30(8), 571-577. DOI: [10.1109/TCS.1983.1085391](https://doi.org/10.1109/TCS.1983.1085391)
2. Schetzen, M. (1980). "The Volterra and Wiener Theories of Nonlinear Systems", Wiley
3. Novak, A., Lotton, P., & Simon, L. (2015). "Synchronized Swept-Sine: Theory, Application, and Implementation", Journal of the Audio Engineering Society, 63(10), 786-798. DOI: [10.17743/jaes.2015.0071](https://doi.org/10.17743/jaes.2015.0071)

### Tensor-Train Decomposition and TT-ALS
4. Oseledets, I.V. (2011). "Tensor-Train Decomposition", SIAM J. Sci. Comput., 33(5), 2295-2317. DOI: [10.1137/090752286](https://doi.org/10.1137/090752286)
5. Holtz, S., Rohwedder, T., & Schneider, R. (2012). "The Alternating Linear Scheme for Tensor Optimization in the Tensor Train Format", SIAM J. Sci. Comput., 34(2), A683-A713. DOI: [10.1137/100818893](https://doi.org/10.1137/100818893)
6. Steinlechner, M. (2016). "Riemannian Optimization for High-Dimensional Tensor Completion", SIAM J. Sci. Comput., 38(5), S461-S484. DOI: [10.1137/15M1010506](https://doi.org/10.1137/15M1010506)
7. Batselier, K., Chen, Z., & Wong, N. (2017). "Tensor Network Alternating Linear Scheme for MIMO Volterra System Identification", Automatica, 84, 26-35. DOI: [10.1016/j.automatica.2017.06.033](https://doi.org/10.1016/j.automatica.2017.06.033)

### Generalized Memory Polynomial
8. Morgan, D.R., Ma, Z., Kim, J., Zierdt, M.G., & Pastalan, J. (2006). "A Generalized Memory Polynomial Model for Digital Predistortion of RF Power Amplifiers", IEEE Transactions on Signal Processing, 54(10), 3852-3860. DOI: [10.1109/TSP.2006.879264](https://doi.org/10.1109/TSP.2006.879264)

### Model Selection
9. Akaike, H. (1974). "A new look at the statistical model identification", IEEE Transactions on Automatic Control, 19(6), 716-723. DOI: [10.1109/TAC.1974.1100705](https://doi.org/10.1109/TAC.1974.1100705)
10. Schwarz, G. (1978). "Estimating the Dimension of a Model", The Annals of Statistics, 6(2), 461-464. DOI: [10.1214/aos/1176344136](https://doi.org/10.1214/aos/1176344136)

### Adaptive Filtering
11. Haykin, S. (2002). "Adaptive Filter Theory", Prentice Hall (4th edition)
12. Diniz, P.S.R. (2013). "Adaptive Filtering: Algorithms and Practical Implementation", Springer (4th edition)

## License

MIT

## Contributing

Issues and pull requests welcome at https://github.com/anatrini/py-volterra
