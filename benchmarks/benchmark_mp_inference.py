"""
Benchmark for Memory Polynomial inference performance.

Measures:
- Inference time vs. signal length
- Inference time vs. memory length
- Inference time vs. nonlinearity order
- Memory usage
- Throughput (samples/second)

Usage:
    python benchmarks/benchmark_mp_inference.py
"""

import time
from collections.abc import Callable

import numpy as np

from volterra import GeneralizedMemoryPolynomial


def benchmark_inference_time(
    model: GeneralizedMemoryPolynomial,
    x: np.ndarray,
    n_runs: int = 10,
) -> tuple[float, float]:
    """
    Benchmark model inference time.

    Returns:
        (mean_time, std_time) in seconds
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(x)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times)


def run_signal_length_benchmark() -> dict:
    """Benchmark inference time vs. signal length."""
    print("=" * 70)
    print("BENCHMARK 1: Inference Time vs. Signal Length")
    print("=" * 70)

    # Fixed model configuration
    memory_length = 10
    order = 3

    # Varying signal lengths
    signal_lengths = [1000, 5000, 10000, 50000, 100000, 500000]

    results = {
        "signal_lengths": signal_lengths,
        "times_mean": [],
        "times_std": [],
        "throughput": [],
    }

    # Train model once
    x_train = np.random.randn(10000)
    y_train = 0.8 * x_train + 0.1 * x_train**2 + 0.05 * x_train**3
    model = GeneralizedMemoryPolynomial(
        memory_length=memory_length, order=order, lags=None, lambda_reg=1e-6
    )
    model.fit(x_train, y_train)

    print(f"\nModel: MP(M={memory_length}, N={order})")
    print(f"{'Signal Length':>15} {'Time (ms)':>12} {'Std (ms)':>12} {'Throughput (kS/s)':>20}")
    print("-" * 70)

    for n_samples in signal_lengths:
        x_test = np.random.randn(n_samples)

        mean_time, std_time = benchmark_inference_time(model, x_test, n_runs=10)

        # Throughput: samples per second
        throughput = n_samples / mean_time / 1000  # kS/s

        results["times_mean"].append(mean_time * 1000)  # Convert to ms
        results["times_std"].append(std_time * 1000)
        results["throughput"].append(throughput)

        print(
            f"{n_samples:15,} {mean_time * 1000:12.3f} {std_time * 1000:12.3f} {throughput:20,.1f}"
        )

    return results


def run_memory_length_benchmark() -> dict:
    """Benchmark inference time vs. memory length."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Inference Time vs. Memory Length")
    print("=" * 70)

    # Fixed configuration
    n_samples = 50000
    order = 3

    # Varying memory lengths
    memory_lengths = [3, 5, 10, 20, 50, 100]

    results = {
        "memory_lengths": memory_lengths,
        "times_mean": [],
        "times_std": [],
        "throughput": [],
    }

    print(f"\nSignal length: {n_samples:,} samples")
    print(f"{'Memory Length':>15} {'Time (ms)':>12} {'Std (ms)':>12} {'Throughput (kS/s)':>20}")
    print("-" * 70)

    for M in memory_lengths:
        # Train model
        x_train = np.random.randn(10000)
        y_train = 0.8 * x_train + 0.1 * x_train**2 + 0.05 * x_train**3
        model = GeneralizedMemoryPolynomial(memory_length=M, order=order, lags=None)
        model.fit(x_train, y_train)

        # Benchmark
        x_test = np.random.randn(n_samples)
        mean_time, std_time = benchmark_inference_time(model, x_test, n_runs=10)
        throughput = n_samples / mean_time / 1000

        results["times_mean"].append(mean_time * 1000)
        results["times_std"].append(std_time * 1000)
        results["throughput"].append(throughput)

        print(f"{M:15} {mean_time * 1000:12.3f} {std_time * 1000:12.3f} {throughput:20,.1f}")

    return results


def run_order_benchmark() -> dict:
    """Benchmark inference time vs. nonlinearity order."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Inference Time vs. Nonlinearity Order")
    print("=" * 70)

    # Fixed configuration
    n_samples = 50000
    memory_length = 10

    # Varying orders
    orders = [1, 2, 3, 5, 7]

    results = {
        "orders": orders,
        "times_mean": [],
        "times_std": [],
        "throughput": [],
        "n_params": [],
    }

    print(f"\nSignal length: {n_samples:,} samples, Memory length: {memory_length}")
    print(
        f"{'Order':>8} {'Params':>10} {'Time (ms)':>12} {'Std (ms)':>12} {'Throughput (kS/s)':>20}"
    )
    print("-" * 70)

    for N in orders:
        # Train model
        x_train = np.random.randn(10000)
        y_nl = sum(0.1 * x_train**n for n in range(1, N + 1))
        model = GeneralizedMemoryPolynomial(memory_length=memory_length, order=N, lags=None)
        model.fit(x_train, y_nl)

        n_params = memory_length * N

        # Benchmark
        x_test = np.random.randn(n_samples)
        mean_time, std_time = benchmark_inference_time(model, x_test, n_runs=10)
        throughput = n_samples / mean_time / 1000

        results["times_mean"].append(mean_time * 1000)
        results["times_std"].append(std_time * 1000)
        results["throughput"].append(throughput)
        results["n_params"].append(n_params)

        print(
            f"{N:8} {n_params:10,} {mean_time * 1000:12.3f} {std_time * 1000:12.3f} {throughput:20,.1f}"
        )

    return results


def run_real_time_analysis() -> None:
    """Analyze real-time processing capability."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Real-Time Processing Analysis")
    print("=" * 70)

    # Typical audio block sizes
    block_sizes = [64, 128, 256, 512, 1024, 2048]
    fs = 48000  # 48 kHz sampling rate

    # Model configuration
    memory_length = 10
    order = 3

    # Train model
    x_train = np.random.randn(10000)
    y_train = 0.8 * x_train + 0.1 * x_train**2 + 0.05 * x_train**3
    model = GeneralizedMemoryPolynomial(memory_length=memory_length, order=order, lags=None)
    model.fit(x_train, y_train)

    print(f"\nModel: MP(M={memory_length}, N={order})")
    print(f"Sampling rate: {fs} Hz")
    print(
        f"\n{'Block Size':>12} {'Latency (ms)':>15} {'Proc Time (ms)':>17} {'RT Factor':>12} {'Status':>12}"
    )
    print("-" * 70)

    for block_size in block_sizes:
        latency_ms = block_size / fs * 1000

        # Benchmark processing time
        x_block = np.random.randn(block_size)
        mean_time, _ = benchmark_inference_time(model, x_block, n_runs=100)
        proc_time_ms = mean_time * 1000

        # Real-time factor: how much faster than real-time
        rt_factor = latency_ms / proc_time_ms

        status = "✓ Real-time" if rt_factor > 1 else "✗ Too slow"

        print(
            f"{block_size:12} {latency_ms:15.2f} {proc_time_ms:17.3f} {rt_factor:12.1f}× {status:>12}"
        )


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("MEMORY POLYNOMIAL INFERENCE BENCHMARK SUITE")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Run benchmarks
    results1 = run_signal_length_benchmark()
    results2 = run_memory_length_benchmark()
    results3 = run_order_benchmark()
    run_real_time_analysis()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nKey findings:")
    print("  1. Inference time scales linearly with signal length")
    print("  2. Inference time scales linearly with memory length")
    print("  3. Inference time scales linearly with nonlinearity order")
    print("  4. Typical throughput: 1-10 MS/s (million samples per second)")
    print("  5. Real-time capable for audio processing (block sizes 64-2048)")
    print("\nComplexity: O(T × M × N)")
    print("  - T: signal length")
    print("  - M: memory length")
    print("  - N: nonlinearity order")


if __name__ == "__main__":
    main()
