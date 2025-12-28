"""
Performance benchmark for Volterra processing up to 5th order.

Measures:
1. Processing latency for different order combinations
2. Throughput in samples/second
3. Real-time factor (how many times faster than real-time)
4. Memory footprint
5. NumPy vs Numba performance comparison
"""

import time
import platform
import sys
import numpy as np

from volterra import (
    VolterraKernelFull,
    VolterraProcessorFull,
    NUMBA_AVAILABLE
)


class VolterraPerformanceBenchmark:
    """Comprehensive performance benchmark suite."""

    def __init__(self):
        self.sample_rate = 48000
        self.results = []

    def print_system_info(self):
        """Display system information."""
        print("="*70)
        print("SYSTEM INFORMATION")
        print("="*70)
        print(f"Platform:     {platform.platform()}")
        print(f"Processor:    {platform.processor()}")
        print(f"Python:       {sys.version.split()[0]}")
        print(f"NumPy:        {np.__version__}")

        if NUMBA_AVAILABLE:
            import numba
            print(f"Numba:        {numba.__version__} (available)")
        else:
            print(f"Numba:        not available")

        print("="*70 + "\n")

    def benchmark_configuration(
        self,
        config_name: str,
        kernel: VolterraKernelFull,
        block_size: int,
        use_numba: bool,
        num_iterations: int = 1000
    ) -> dict:
        """
        Benchmark a single configuration.

        Returns:
            Dictionary with performance metrics
        """
        proc = VolterraProcessorFull(kernel, sample_rate=self.sample_rate, use_numba=use_numba)

        # Generate test signal
        x_test = np.random.randn(block_size).astype(np.float64) * 0.1

        # Warmup (important for JIT compilation)
        for _ in range(20):
            _ = proc.process_block(x_test)

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            y = proc.process_block(x_test)
        end = time.perf_counter()

        elapsed = end - start
        latency_per_block_ms = (elapsed / num_iterations) * 1000
        samples_per_second = (block_size * num_iterations) / elapsed

        # Real-time factor: how much faster than real-time playback
        real_time_duration_ms = (block_size / self.sample_rate) * 1000
        realtime_factor = real_time_duration_ms / latency_per_block_ms

        # Memory footprint
        memory_kb = kernel.estimate_memory_bytes() / 1024

        return {
            'config': config_name,
            'block_size': block_size,
            'engine': 'Numba' if use_numba and NUMBA_AVAILABLE else 'NumPy',
            'latency_ms': latency_per_block_ms,
            'samples_per_sec': samples_per_second,
            'realtime_factor': realtime_factor,
            'memory_kb': memory_kb,
            'max_order': kernel.max_order
        }

    def run_full_benchmark(self):
        """Run comprehensive benchmark across all configurations."""
        print("="*70)
        print("VOLTERRA PERFORMANCE BENCHMARK")
        print("="*70 + "\n")

        kernel_length = 512
        block_sizes = [128, 256, 512, 1024]

        # Generate test kernels
        h1 = np.random.randn(kernel_length).astype(np.float64) * 0.1
        h2 = np.random.randn(kernel_length).astype(np.float64) * 0.05
        h3 = np.random.randn(kernel_length).astype(np.float64) * 0.01
        h4 = np.random.randn(kernel_length).astype(np.float64) * 0.005
        h5 = np.random.randn(kernel_length).astype(np.float64) * 0.002

        # Kernel configurations to test
        kernel_configs = [
            ("h1 only", VolterraKernelFull(h1=h1)),
            ("h1+h2", VolterraKernelFull(h1=h1, h2=h2, h2_is_diagonal=True)),
            ("h1+h2+h3", VolterraKernelFull(h1=h1, h2=h2, h3_diagonal=h3, h2_is_diagonal=True)),
            ("h1+h2+h3+h4", VolterraKernelFull(h1=h1, h2=h2, h3_diagonal=h3, h4_diagonal=h4, h2_is_diagonal=True)),
            ("h1+h2+h3+h4+h5", VolterraKernelFull(h1=h1, h2=h2, h3_diagonal=h3, h4_diagonal=h4, h5_diagonal=h5, h2_is_diagonal=True)),
        ]

        # Test both engines if Numba available
        engines = [False]
        if NUMBA_AVAILABLE:
            engines.append(True)

        for config_name, kernel in kernel_configs:
            print(f"\n{config_name} (order {kernel.max_order})")
            print("-"*70)

            for use_numba in engines:
                engine_name = "Numba" if use_numba else "NumPy"

                for block_size in block_sizes:
                    result = self.benchmark_configuration(
                        config_name, kernel, block_size, use_numba, num_iterations=500
                    )

                    self.results.append(result)

                    # Format output
                    status = "✓" if result['realtime_factor'] > 1 else "✗"
                    print(f"  [{engine_name:5s}] Block={block_size:4d}: "
                          f"{result['latency_ms']:6.3f} ms/block, "
                          f"{result['realtime_factor']:5.1f}× RT {status}")

        self._print_summary()
        return self.results

    def _print_summary(self):
        """Print benchmark summary and recommendations."""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)

        # Find best configurations
        realtime_configs = [r for r in self.results if r['realtime_factor'] > 1]

        if not realtime_configs:
            print("\n⚠ WARNING: No configuration achieved real-time performance!")
            print("   Consider:")
            print("   - Installing Numba: pip install numba")
            print("   - Using larger block sizes")
            print("   - Reducing kernel orders")
            return

        # Best overall (highest order real-time)
        best_overall = max(realtime_configs, key=lambda r: (r['max_order'], r['realtime_factor']))

        print(f"\n🎯 RECOMMENDED CONFIGURATION:")
        print(f"   Orders:       {best_overall['config']}")
        print(f"   Engine:       {best_overall['engine']}")
        print(f"   Block size:   {best_overall['block_size']} samples")
        print(f"   Latency:      {best_overall['latency_ms']:.3f} ms/block "
              f"({best_overall['block_size']/48:.2f} ms buffer)")
        print(f"   Performance:  {best_overall['realtime_factor']:.1f}× real-time")
        print(f"   Memory:       {best_overall['memory_kb']:.1f} KB")

        # Fastest processing
        fastest = min(realtime_configs, key=lambda r: r['latency_ms'])
        if fastest != best_overall:
            print(f"\n⚡ FASTEST CONFIGURATION:")
            print(f"   {fastest['config']} ({fastest['engine']}, block={fastest['block_size']})")
            print(f"   Latency: {fastest['latency_ms']:.3f} ms/block")

        # Engine comparison if both available
        if NUMBA_AVAILABLE:
            print("\n📊 ENGINE COMPARISON (block_size=512):")
            print(f"{'Config':<20} {'NumPy (ms)':<12} {'Numba (ms)':<12} {'Speedup'}")
            print("-"*60)

            configs_512 = {}
            for r in self.results:
                if r['block_size'] == 512:
                    key = r['config']
                    if key not in configs_512:
                        configs_512[key] = {}
                    configs_512[key][r['engine']] = r['latency_ms']

            for config, engines in configs_512.items():
                if 'NumPy' in engines and 'Numba' in engines:
                    speedup = engines['NumPy'] / engines['Numba']
                    print(f"{config:<20} {engines['NumPy']:>10.3f}   "
                          f"{engines['Numba']:>10.3f}   {speedup:>6.1f}×")

        print("\n" + "="*70)

    def run_quick_test(self):
        """Quick test for basic performance check."""
        print("="*70)
        print("QUICK PERFORMANCE TEST")
        print("="*70 + "\n")

        # Most demanding config: all orders
        kernel = VolterraKernelFull.from_polynomial_coeffs(
            N=512, a1=0.9, a2=0.1, a3=0.03, a4=0.01, a5=0.02
        )

        block_size = 512

        # Test both engines
        for use_numba in [False, True]:
            if not use_numba or NUMBA_AVAILABLE:
                result = self.benchmark_configuration(
                    "h1+h2+h3+h4+h5", kernel, block_size, use_numba, num_iterations=100
                )

                engine = "Numba" if use_numba else "NumPy"
                status = "✓ Real-time capable" if result['realtime_factor'] > 1 else "✗ Not real-time"

                print(f"{engine} Engine:")
                print(f"  Latency:         {result['latency_ms']:.3f} ms/block")
                print(f"  Real-time factor: {result['realtime_factor']:.1f}×")
                print(f"  Status:          {status}")
                print()


def demo_progressive_orders():
    """Demonstrate how performance scales with order."""
    print("="*70)
    print("PERFORMANCE SCALING BY ORDER")
    print("="*70 + "\n")

    N = 512
    block_size = 512

    configs = [
        (1, VolterraKernelFull.from_polynomial_coeffs(N, a1=1.0)),
        (2, VolterraKernelFull.from_polynomial_coeffs(N, a1=1.0, a2=0.1)),
        (3, VolterraKernelFull.from_polynomial_coeffs(N, a1=1.0, a2=0.1, a3=0.03)),
        (4, VolterraKernelFull.from_polynomial_coeffs(N, a1=1.0, a2=0.1, a3=0.03, a4=0.01)),
        (5, VolterraKernelFull.from_polynomial_coeffs(N, a1=1.0, a2=0.1, a3=0.03, a4=0.01, a5=0.02)),
    ]

    print(f"Block size: {block_size}, Kernel length: {N}")
    print(f"\n{'Order':<10} {'Latency (ms)':<15} {'RT Factor':<12} {'Status'}")
    print("-"*60)

    use_numba = NUMBA_AVAILABLE

    for order, kernel in configs:
        proc = VolterraProcessorFull(kernel, use_numba=use_numba)
        x = np.random.randn(block_size) * 0.1

        # Warmup
        for _ in range(10):
            proc.process_block(x)

        # Measure
        start = time.perf_counter()
        for _ in range(100):
            y = proc.process_block(x)
        elapsed = time.perf_counter() - start

        latency_ms = (elapsed / 100) * 1000
        rt_factor = (block_size / 48000 * 1000) / latency_ms
        status = "✓" if rt_factor > 1 else "✗"

        print(f"{order:<10} {latency_ms:>13.3f}   {rt_factor:>10.1f}×   {status}")

    engine_name = "Numba" if use_numba else "NumPy"
    print(f"\nEngine: {engine_name}")
    print("="*70)


if __name__ == "__main__":
    benchmark = VolterraPerformanceBenchmark()

    # Print system info
    benchmark.print_system_info()

    # Run quick test
    benchmark.run_quick_test()

    # Show scaling
    demo_progressive_orders()

    # Full benchmark (uncomment for complete analysis)
    print("\nRunning full benchmark suite...")
    print("(This may take a few minutes)")
    benchmark.run_full_benchmark()
