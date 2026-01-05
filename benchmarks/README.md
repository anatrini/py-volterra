# Benchmarks

This directory contains performance benchmarks for py-volterra models.

## Available Benchmarks

### 1. Memory Polynomial Inference (`benchmark_mp_inference.py`)

Measures MP inference performance across different configurations.

**Benchmarks:**
- Inference time vs. signal length
- Inference time vs. memory length
- Inference time vs. nonlinearity order
- Real-time processing capability (audio block sizes)

**Run:**
```bash
python benchmarks/benchmark_mp_inference.py
```

**Expected output:**
- Throughput: 1-10 MS/s (million samples per second)
- Real-time capable for audio (48 kHz)
- Linear scaling with all parameters

---

### 2. Generalized Memory Polynomial Inference (`benchmark_gmp_inference.py`)

Compares GMP configurations and overhead vs. standard MP.

**Benchmarks:**
- Different lag structures (Light, Medium, Heavy, Full)
- Inference time vs. number of cross-lags
- MP vs. GMP computational overhead

**Run:**
```bash
python benchmarks/benchmark_gmp_inference.py
```

**Expected findings:**
- GMP overhead: 10-50% depending on lag structure
- Light GMP: ~10-15% overhead
- Heavy GMP: ~40-50% overhead
- Overhead scales with cross-lag count

---

### 3. Tensor-Train Fit Performance (`benchmark_tt_fit.py`)

Measures TT-Volterra fit time and compression ratios.

**Benchmarks:**
- Fit time vs. TT rank (quadratic scaling)
- Fit time vs. signal length (linear scaling)
- Fit time vs. memory length
- MIMO scaling (inputs × outputs)
- Compression ratio analysis

**Run:**
```bash
python benchmarks/benchmark_tt_fit.py
```

**Expected findings:**
- Fit time scales quadratically with rank
- Compression: 10-1000× vs. full Volterra
- Rank 2-3 recommended for most applications
- MIMO fit scales with (I + O) × iterations

---

### 4. Model Comparison (`benchmark_model_comparison.py`)

Comprehensive comparison of all model types.

**Benchmarks:**
- Diagonal system (MP optimal)
- Cross-memory system (GMP optimal)
- Scaling comparison across all models

**Metrics:**
- Fit time
- Inference time
- Parameter count
- Model accuracy (NMSE)

**Run:**
```bash
python benchmarks/benchmark_model_comparison.py
```

**Expected findings:**
- MP fastest for diagonal systems
- GMP best accuracy/speed trade-off for cross-memory
- TT-Full essential for high-dimensional MIMO

---

## Running All Benchmarks

```bash
# Run all benchmarks sequentially
python benchmarks/benchmark_mp_inference.py
python benchmarks/benchmark_gmp_inference.py
python benchmarks/benchmark_tt_fit.py
python benchmarks/benchmark_model_comparison.py
```

Or use a shell script:
```bash
for script in benchmarks/benchmark_*.py; do
    echo "Running $script..."
    python "$script"
    echo ""
done
```

---

## Interpreting Results

### Throughput Metrics

- **MS/s (Mega-Samples/second)**: Million samples processed per second
- **kS/s (Kilo-Samples/second)**: Thousand samples processed per second

**Typical values:**
- MP inference: 1-10 MS/s
- GMP inference: 0.5-5 MS/s
- TT inference: 0.5-3 MS/s

### Real-Time Factors

- **RT Factor > 1**: Real-time capable
- **RT Factor >> 10**: Plenty of headroom for additional processing
- **RT Factor < 1**: Too slow for real-time

### NMSE (Normalized Mean Squared Error)

- **< -30 dB**: Excellent fit
- **-20 to -30 dB**: Good fit
- **-10 to -20 dB**: Acceptable fit
- **> -10 dB**: Poor fit

---

## Hardware Considerations

Benchmark results depend on:
- **CPU**: Single-core performance matters most
- **RAM**: Minimal impact (models are small)
- **Python**: CPython 3.10+ recommended
- **NumPy**: Built with optimized BLAS (MKL, OpenBLAS)

**To check NumPy configuration:**
```python
import numpy as np
np.show_config()
```

---

## Performance Optimization

### For Faster Inference:
1. Install `numba` for JIT compilation (10× speedup for high orders)
2. Use smaller `memory_length` if acceptable
3. Reduce `order` if higher orders don't improve accuracy
4. Prefer MP over GMP if no cross-memory effects

### For Faster Fitting:
1. Reduce `max_iter` for TT-ALS (20-30 usually sufficient)
2. Use smaller training datasets (5K-20K samples often enough)
3. Lower TT ranks (r=2-3 recommended)
4. Use `lambda_reg > 0` for faster convergence

---

## Benchmark Environment

Results in this README were obtained on:
- CPU: (your CPU here)
- RAM: (your RAM here)
- Python: 3.10+
- NumPy: (with MKL/OpenBLAS)
- OS: (your OS here)

**Note:** Actual performance will vary based on hardware and system configuration.

---

## Citation

If you use these benchmarks in research, please cite:

```bibtex
@software{py_volterra,
  title={py-volterra: Tensor-Train Volterra System Identification},
  author={anatrini},
  year={2026},
  url={https://github.com/anatrini/py-volterra}
}
```

---

## See Also

- [Main README](../README.md)
- [Notebooks](../notebooks/)
- [Examples](../examples/)
- [Tests](../tests/)
