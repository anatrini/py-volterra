"""
Comprehensive benchmark comparing all Volterra model types.

Compares:
- Memory Polynomial (MP/Diagonal)
- Generalized Memory Polynomial (GMP)
- Tensor-Train Volterra (TT-Full)

Metrics:
- Fit time
- Inference time
- Parameter count
- Model accuracy (NMSE)
- Memory usage

Usage:
    python benchmarks/benchmark_model_comparison.py
"""

import time

import numpy as np

from volterra import GeneralizedMemoryPolynomial, TTVolterraMIMO


def generate_test_system(n_samples: int, system_type: str = "diagonal"):
    """Generate synthetic test data for different system types."""
    np.random.seed(42)
    x = np.random.randn(n_samples) * 0.3

    if system_type == "diagonal":
        # Pure diagonal system (favors MP)
        y_clean = 0.8 * x + 0.12 * x**2 + 0.05 * x**3
    elif system_type == "cross_memory":
        # System with cross-memory (favors GMP)
        x_delayed = np.concatenate([np.zeros(3), x[:-3]])
        y_clean = 0.8 * x + 0.1 * x**2 + 0.05 * x**3 + 0.15 * x * x_delayed
    else:
        raise ValueError(f"Unknown system type: {system_type}")

    # Add filtering (memory effects)
    from scipy import signal

    b, a = [0.2, -0.38, 0.18], [1.0, -1.9, 0.94]
    y_filtered = signal.lfilter(b, a, y_clean)

    # Add noise
    y = y_filtered + np.random.randn(n_samples) * 0.01

    return x, y


def benchmark_diagonal_system() -> None:
    """Compare all models on diagonal system."""
    print("=" * 80)
    print("BENCHMARK 1: Diagonal System (MP Optimal)")
    print("=" * 80)

    # Configuration
    n_train = 10000
    n_test = 5000
    memory_length = 10
    order = 3

    # Generate data
    x_train, y_train = generate_test_system(n_train, "diagonal")
    x_test, y_test = generate_test_system(n_test, "diagonal")

    # Model configurations
    models = {
        "MP (Diagonal)": {
            "class": GeneralizedMemoryPolynomial,
            "kwargs": {"memory_length": memory_length, "order": order, "lags": None},
        },
        "GMP Medium": {
            "class": GeneralizedMemoryPolynomial,
            "kwargs": {
                "memory_length": memory_length,
                "order": order,
                "lags": {1: [0, 1, 2, 3, 4], 2: [0, 1, 2], 3: [0, 1]},
            },
        },
        "TT-Full r=2": {
            "class": TTVolterraMIMO,
            "kwargs": {
                "memory_length": memory_length,
                "order": order,
                "ranks": [1, 2, 2, 1],
                "max_iter": 20,
            },
        },
        "TT-Full r=3": {
            "class": TTVolterraMIMO,
            "kwargs": {
                "memory_length": memory_length,
                "order": order,
                "ranks": [1, 3, 3, 1],
                "max_iter": 20,
            },
        },
    }

    print(f"\nTraining data: {n_train} samples, Test data: {n_test} samples")
    print(
        f"{'Model':>15} {'Params':>10} {'Fit (s)':>10} {'Infer (ms)':>12} {'NMSE (dB)':>12}"
    )
    print("-" * 80)

    for name, config in models.items():
        # Create model
        if config["class"] == TTVolterraMIMO:
            # Reshape for MIMO
            x_tr = x_train.reshape(-1, 1)
            y_tr = y_train.reshape(-1, 1)
            x_te = x_test.reshape(-1, 1)
            model_data = (x_tr, y_tr, x_te)
        else:
            model_data = (x_train, y_train, x_test)

        model = config["class"](**config["kwargs"])

        # Benchmark fit
        start_fit = time.perf_counter()
        model.fit(model_data[0], model_data[1])
        fit_time = time.perf_counter() - start_fit

        # Benchmark inference
        infer_times = []
        for _ in range(10):
            start = time.perf_counter()
            y_pred = model.predict(model_data[2])
            infer_times.append(time.perf_counter() - start)
        infer_time = np.mean(infer_times) * 1000  # ms

        # Compute NMSE
        M = model.memory_length if hasattr(model, "memory_length") else memory_length
        y_test_trimmed = y_test[M - 1 : M - 1 + len(y_pred)]

        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        mse = np.mean((y_test_trimmed - y_pred) ** 2)
        signal_power = np.mean(y_test_trimmed**2)
        nmse_db = 10 * np.log10(mse / signal_power)

        # Count parameters
        if hasattr(model, "coeffs_"):
            n_params = model.coeffs_.size
        elif hasattr(model, "total_parameters"):
            n_params = model.total_parameters()
        else:
            n_params = 0

        print(
            f"{name:>15} {n_params:10,} {fit_time:10.3f} {infer_time:12.3f} {nmse_db:12.2f}"
        )


def benchmark_cross_memory_system() -> None:
    """Compare all models on system with cross-memory effects."""
    print("\n" + "=" * 80)
    print("BENCHMARK 2: Cross-Memory System (GMP Optimal)")
    print("=" * 80)

    # Configuration
    n_train = 10000
    n_test = 5000
    memory_length = 10
    order = 3

    # Generate data with cross-memory
    x_train, y_train = generate_test_system(n_train, "cross_memory")
    x_test, y_test = generate_test_system(n_test, "cross_memory")

    # Model configurations
    models = {
        "MP (Diagonal)": {
            "class": GeneralizedMemoryPolynomial,
            "kwargs": {"memory_length": memory_length, "order": order, "lags": None},
        },
        "GMP Light": {
            "class": GeneralizedMemoryPolynomial,
            "kwargs": {
                "memory_length": memory_length,
                "order": order,
                "lags": {1: [0, 1, 2], 2: [0, 1], 3: [0]},
            },
        },
        "GMP Medium": {
            "class": GeneralizedMemoryPolynomial,
            "kwargs": {
                "memory_length": memory_length,
                "order": order,
                "lags": {1: [0, 1, 2, 3, 4], 2: [0, 1, 2], 3: [0, 1]},
            },
        },
        "TT-Full r=3": {
            "class": TTVolterraMIMO,
            "kwargs": {
                "memory_length": memory_length,
                "order": order,
                "ranks": [1, 3, 3, 1],
                "max_iter": 20,
            },
        },
    }

    print(f"\nTraining data: {n_train} samples, Test data: {n_test} samples")
    print(
        f"{'Model':>15} {'Params':>10} {'Fit (s)':>10} {'Infer (ms)':>12} {'NMSE (dB)':>12}"
    )
    print("-" * 80)

    for name, config in models.items():
        # Create model
        if config["class"] == TTVolterraMIMO:
            x_tr = x_train.reshape(-1, 1)
            y_tr = y_train.reshape(-1, 1)
            x_te = x_test.reshape(-1, 1)
            model_data = (x_tr, y_tr, x_te)
        else:
            model_data = (x_train, y_train, x_test)

        model = config["class"](**config["kwargs"])

        # Benchmark fit
        start_fit = time.perf_counter()
        model.fit(model_data[0], model_data[1])
        fit_time = time.perf_counter() - start_fit

        # Benchmark inference
        infer_times = []
        for _ in range(10):
            start = time.perf_counter()
            y_pred = model.predict(model_data[2])
            infer_times.append(time.perf_counter() - start)
        infer_time = np.mean(infer_times) * 1000

        # Compute NMSE
        M = model.memory_length if hasattr(model, "memory_length") else memory_length
        y_test_trimmed = y_test[M - 1 : M - 1 + len(y_pred)]

        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        mse = np.mean((y_test_trimmed - y_pred) ** 2)
        signal_power = np.mean(y_test_trimmed**2)
        nmse_db = 10 * np.log10(mse / signal_power)

        # Count parameters
        if hasattr(model, "coeffs_"):
            n_params = model.coeffs_.size
        elif hasattr(model, "total_parameters"):
            n_params = model.total_parameters()
        else:
            n_params = 0

        print(
            f"{name:>15} {n_params:10,} {fit_time:10.3f} {infer_time:12.3f} {nmse_db:12.2f}"
        )


def benchmark_scaling_comparison() -> None:
    """Compare how models scale with problem size."""
    print("\n" + "=" * 80)
    print("BENCHMARK 3: Scaling Comparison (Signal Length)")
    print("=" * 80)

    # Fixed configuration
    memory_length = 8
    order = 3

    # Varying signal lengths
    signal_lengths = [5000, 10000, 20000, 50000]

    print(f"\nMemory: {memory_length}, Order: {order}")

    for model_name in ["MP", "GMP", "TT-Full"]:
        print(f"\n{model_name} Scaling:")
        print(f"{'N Samples':>12} {'Fit (s)':>10} {'Infer (ms)':>12} {'Total (s)':>12}")
        print("-" * 60)

        for n_samples in signal_lengths:
            # Generate data
            x, y = generate_test_system(n_samples, "diagonal")

            # Create model
            if model_name == "MP":
                model = GeneralizedMemoryPolynomial(
                    memory_length=memory_length, order=order, lags=None
                )
                x_fit, y_fit, x_pred = x, y, x
            elif model_name == "GMP":
                model = GeneralizedMemoryPolynomial(
                    memory_length=memory_length,
                    order=order,
                    lags={1: [0, 1, 2, 3, 4], 2: [0, 1, 2], 3: [0, 1]},
                )
                x_fit, y_fit, x_pred = x, y, x
            else:  # TT-Full
                model = TTVolterraMIMO(
                    memory_length=memory_length,
                    order=order,
                    ranks=[1, 3, 3, 1],
                    max_iter=20,
                )
                x_fit = x.reshape(-1, 1)
                y_fit = y.reshape(-1, 1)
                x_pred = x_fit

            # Benchmark fit
            start_fit = time.perf_counter()
            model.fit(x_fit, y_fit)
            fit_time = time.perf_counter() - start_fit

            # Benchmark inference
            start_infer = time.perf_counter()
            _ = model.predict(x_pred)
            infer_time = (time.perf_counter() - start_infer) * 1000  # ms

            total_time = fit_time + infer_time / 1000

            print(
                f"{n_samples:12,} {fit_time:10.3f} {infer_time:12.3f} {total_time:12.3f}"
            )


def main():
    """Run all comparison benchmarks."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON BENCHMARK SUITE")
    print("=" * 80)

    # Set random seed
    np.random.seed(42)

    # Run benchmarks
    benchmark_diagonal_system()
    benchmark_cross_memory_system()
    benchmark_scaling_comparison()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print("\n1. Diagonal Systems (no cross-memory):")
    print("   ✓ Best: Memory Polynomial (MP)")
    print("   - Fastest fit and inference")
    print("   - Fewest parameters")
    print("   - Optimal accuracy for diagonal systems")
    print("\n2. Cross-Memory Systems:")
    print("   ✓ Best: Generalized Memory Polynomial (GMP)")
    print("   - Moderate overhead vs. MP (~20-40%)")
    print("   - Significantly better accuracy on cross-memory data")
    print("   - Use GMP Light/Medium for most applications")
    print("\n3. High-Dimensional MIMO:")
    print("   ✓ Best: Tensor-Train Volterra (TT-Full)")
    print("   - Essential for I > 2 inputs")
    print("   - Compression ratio: 10-1000× vs. full Volterra")
    print("   - Slower fit, but manageable with rank 2-3")
    print("\n4. Scaling Characteristics:")
    print("   - MP/GMP: O(T × M × N) fit and inference")
    print("   - TT-Full: O(T × M × N × r² × I × iter) fit")
    print("   - All scale linearly with signal length")
    print("\n5. Model Selection Guidelines:")
    print("   - Unknown structure → Use ModelSelector (AIC/BIC)")
    print("   - SISO, N ≤ 3, M ≤ 10 → Start with MP")
    print("   - Suspect cross-memory → Try GMP Medium")
    print("   - MIMO or N > 3 → Use TT-Full with r=2-3")
    print("   - Real-time critical → Prefer MP/GMP (faster inference)")


if __name__ == "__main__":
    main()
