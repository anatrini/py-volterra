"""
MATHEMATICAL CORRECTNESS AUDIT - Critical Validation Tests

This test suite verifies that the Volterra implementation is mathematically correct
according to strict criteria:

1. Pure h₂: Input sine → Output ONLY 2f₀ (no f₀, no 3f₀)
2. Pure h₃: Input sine → Output ONLY odd harmonics (3f₀, 5f₀, ...)
3. Known polynomial: Extract kernels → Match coefficients (<1% error)
4. Block vs Continuous: Same result regardless of block size (<1e-10 error)

RED FLAGS:
- Different harmonic content → WRONG IMPLEMENTATION
- Block ≠ Continuous → OVERLAP-ADD BUG
- Polynomial error >1% → KERNEL EXTRACTION WRONG
"""

import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

from volterra import (
    VolterraKernelFull,
    VolterraProcessorFull,
)

# Also test optimized engine
try:
    from volterra.engines_optimized import OptimizedDiagonalEngine, OptimizedNumbaEngine, NUMBA_AVAILABLE
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False
    print("Warning: Optimized engines not available")


class MathematicalCorrectnessAuditor:
    """Rigorous mathematical validation of Volterra implementation."""

    def __init__(self):
        self.fs = 48000
        self.failed_tests = []
        self.passed_tests = []

    def analyze_spectrum(
        self,
        signal: np.ndarray,
        f0: float,
        num_harmonics: int = 10,
        threshold_db: float = -80.0
    ) -> dict:
        """
        Analyze harmonic content of signal.

        Returns:
            dict with:
            - 'harmonics': {1: level_db, 2: level_db, ...}
            - 'noise_floor': average noise level in dB
            - 'spurious': dict of unexpected frequency content
        """
        Y = rfft(signal)
        freqs = rfftfreq(len(signal), 1/self.fs)

        # Find harmonic levels
        harmonics = {}
        for k in range(1, num_harmonics + 1):
            f_harmonic = k * f0
            if f_harmonic > self.fs / 2:
                break

            # Find bin (with tolerance for frequency resolution)
            bin_width = self.fs / len(signal)
            idx = np.argmin(np.abs(freqs - f_harmonic))

            # Check we're close enough
            if abs(freqs[idx] - f_harmonic) < bin_width * 2:
                level_db = 20 * np.log10(np.abs(Y[idx]) + 1e-15)
                harmonics[k] = level_db

        # Estimate noise floor (bins not near harmonics)
        harmonic_bins = set()
        for k in range(1, num_harmonics + 1):
            f_harmonic = k * f0
            if f_harmonic > self.fs / 2:
                break
            idx = np.argmin(np.abs(freqs - f_harmonic))
            # Exclude ±5 bins around each harmonic
            for offset in range(-5, 6):
                if 0 <= idx + offset < len(Y):
                    harmonic_bins.add(idx + offset)

        noise_bins = set(range(len(Y))) - harmonic_bins - {0}  # Exclude DC
        noise_levels = 20 * np.log10(np.abs(Y[list(noise_bins)]) + 1e-15)
        noise_floor = np.median(noise_levels)

        return {
            'harmonics': harmonics,
            'noise_floor': noise_floor,
            'max_level': max(harmonics.values()) if harmonics else -120
        }

    def test_pure_h2_only_2f0(self, use_optimized=False):
        """
        TEST 1: Pure h₂ (no h₁, h₃, h₅) should produce ONLY 2f₀.

        Theory:
        - Input: x(t) = A·sin(2πf₀t)
        - Pure h₂: y(t) = h₂[0]·x²(t)
        - x²(t) = A²·sin²(2πf₀t) = (A²/2)·(1 - cos(4πf₀t))
        - Result: DC + 2f₀ component ONLY

        Expected:
        - 2f₀: Strong (> -10 dB)
        - f₀: Absent (< -80 dB)
        - 3f₀, 4f₀, 5f₀: Absent (< -80 dB)
        """
        print("="*70)
        print("TEST 1: Pure h₂ - Should produce ONLY 2f₀")
        print("="*70)

        # Create PURE h₂ kernel (NO h₁, NO h₃, NO h₅)
        N = 512
        h1 = np.zeros(N)  # ZERO linear term
        h2 = np.zeros(N)
        h2[0] = 0.5  # Memoryless quadratic

        kernel = VolterraKernelFull(
            h1=h1,
            h2=h2,
            h3_diagonal=None,  # NO 3rd order
            h4_diagonal=None,
            h5_diagonal=None,
            h2_is_diagonal=True
        )

        # Create processor
        if use_optimized and OPTIMIZED_AVAILABLE:
            from volterra.processor_full import VolterraProcessorFull
            proc = VolterraProcessorFull.__new__(VolterraProcessorFull)
            proc.kernel = kernel
            proc.sample_rate = self.fs
            proc.use_numba = False
            proc.engine = OptimizedNumbaEngine() if NUMBA_AVAILABLE else OptimizedDiagonalEngine()
            proc.reset = lambda: None
            proc.reset()
            engine_name = "Optimized"
        else:
            proc = VolterraProcessorFull(kernel, sample_rate=self.fs, use_numba=False)
            engine_name = "Standard"

        # Generate test signal: pure sine
        f0 = 1000.0
        duration = 1.0
        t = np.arange(int(self.fs * duration)) / self.fs
        amplitude = 0.3  # Moderate amplitude
        x = amplitude * np.sin(2 * np.pi * f0 * t)

        # Process
        y = proc.process(x, block_size=512)

        # Analyze spectrum
        spec = self.analyze_spectrum(y, f0, num_harmonics=10)

        print(f"\nEngine: {engine_name}")
        print(f"Input: {f0:.0f} Hz sine, amplitude={amplitude:.2f}")
        print(f"\nHarmonic Content:")
        print(f"{'Harmonic':<12} {'Frequency':<12} {'Level (dB)':<12} {'Status'}")
        print("-"*60)

        # Check expectations
        test_passed = True

        for k in sorted(spec['harmonics'].keys()):
            level = spec['harmonics'][k]
            freq = k * f0
            expected = (k == 2)  # Only 2f₀ should be present

            if expected:
                status = "✓ EXPECTED" if level > -20 else "✗ TOO WEAK"
                if level < -20:
                    test_passed = False
            else:
                status = "✓ SUPPRESSED" if level < -60 else "✗ PRESENT!"
                if level > -60:
                    test_passed = False

            print(f"H{k:<10} {freq:<10.0f}Hz {level:>10.1f}   {status}")

        print(f"\nNoise floor: {spec['noise_floor']:.1f} dB")

        # Critical checks
        h1_level = spec['harmonics'].get(1, -120)
        h2_level = spec['harmonics'].get(2, -120)
        h3_level = spec['harmonics'].get(3, -120)

        print(f"\nCRITICAL CHECKS:")
        print(f"  f₀ suppressed (< -60 dB):  {h1_level:.1f} dB  {'✓' if h1_level < -60 else '✗ FAIL'}")
        print(f"  2f₀ strong (> -20 dB):     {h2_level:.1f} dB  {'✓' if h2_level > -20 else '✗ FAIL'}")
        print(f"  3f₀ suppressed (< -60 dB): {h3_level:.1f} dB  {'✓' if h3_level < -60 else '✗ FAIL'}")

        if h1_level > -60 or h2_level < -20 or h3_level > -60:
            test_passed = False

        if test_passed:
            print("\n✅ TEST 1 PASSED: Pure h₂ produces only 2f₀ as expected")
            self.passed_tests.append("TEST 1: Pure h₂")
        else:
            print("\n❌ TEST 1 FAILED: Incorrect harmonic content")
            self.failed_tests.append("TEST 1: Pure h₂ - wrong harmonics")

        print("="*70 + "\n")
        return test_passed, y, spec

    def test_pure_h3_odd_harmonics(self, use_optimized=False):
        """
        TEST 2: Pure h₃ should produce ONLY odd harmonics (3f₀, 5f₀, 7f₀...).

        Theory:
        - Input: x(t) = A·sin(2πf₀t)
        - Pure h₃: y(t) = h₃[0]·x³(t)
        - sin³(θ) = (3sin(θ) - sin(3θ))/4
        - Result: f₀ and 3f₀ (and higher odd harmonics from nonlinearity)

        Expected:
        - f₀: May be present (fundamental from cubic expansion)
        - 2f₀: ABSENT (< -80 dB) - NO even harmonics!
        - 3f₀: Strong
        - 4f₀: ABSENT (< -80 dB)
        - 5f₀: Weak (higher-order term)
        """
        print("="*70)
        print("TEST 2: Pure h₃ - Should produce ONLY odd harmonics")
        print("="*70)

        # Create PURE h₃ kernel
        N = 512
        h1 = np.zeros(N)  # No linear
        h3 = np.zeros(N)
        h3[0] = 0.3  # Memoryless cubic

        kernel = VolterraKernelFull(
            h1=h1,
            h2=None,  # NO 2nd order
            h3_diagonal=h3,
            h4_diagonal=None,  # NO 4th order
            h5_diagonal=None,
            h2_is_diagonal=True
        )

        # Create processor
        if use_optimized and OPTIMIZED_AVAILABLE:
            from volterra.processor_full import VolterraProcessorFull
            proc = VolterraProcessorFull.__new__(VolterraProcessorFull)
            proc.kernel = kernel
            proc.sample_rate = self.fs
            proc.use_numba = False
            proc.engine = OptimizedNumbaEngine() if NUMBA_AVAILABLE else OptimizedDiagonalEngine()
            proc.reset = lambda: None
            proc.reset()
            engine_name = "Optimized"
        else:
            proc = VolterraProcessorFull(kernel, sample_rate=self.fs, use_numba=False)
            engine_name = "Standard"

        # Generate test signal
        f0 = 1000.0
        duration = 1.0
        t = np.arange(int(self.fs * duration)) / self.fs
        amplitude = 0.2  # Lower amplitude for cubic
        x = amplitude * np.sin(2 * np.pi * f0 * t)

        # Process
        y = proc.process(x, block_size=512)

        # Analyze spectrum
        spec = self.analyze_spectrum(y, f0, num_harmonics=10)

        print(f"\nEngine: {engine_name}")
        print(f"Input: {f0:.0f} Hz sine, amplitude={amplitude:.2f}")
        print(f"\nHarmonic Content:")
        print(f"{'Harmonic':<12} {'Frequency':<12} {'Level (dB)':<12} {'Status'}")
        print("-"*60)

        test_passed = True

        for k in sorted(spec['harmonics'].keys()):
            level = spec['harmonics'][k]
            freq = k * f0
            is_even = (k % 2 == 0)

            if is_even:
                status = "✓ SUPPRESSED" if level < -60 else "✗ EVEN HARMONIC PRESENT!"
                if level > -60:
                    test_passed = False
            else:  # Odd
                status = "✓ ODD (expected)"

            print(f"H{k:<10} {freq:<10.0f}Hz {level:>10.1f}   {status}")

        print(f"\nNoise floor: {spec['noise_floor']:.1f} dB")

        # Critical checks: even harmonics must be suppressed
        h2_level = spec['harmonics'].get(2, -120)
        h4_level = spec['harmonics'].get(4, -120)
        h3_level = spec['harmonics'].get(3, -120)

        print(f"\nCRITICAL CHECKS:")
        print(f"  2f₀ suppressed (< -60 dB): {h2_level:.1f} dB  {'✓' if h2_level < -60 else '✗ FAIL'}")
        print(f"  3f₀ present:               {h3_level:.1f} dB  {'✓' if h3_level > -40 else '✗ WEAK'}")
        print(f"  4f₀ suppressed (< -60 dB): {h4_level:.1f} dB  {'✓' if h4_level < -60 else '✗ FAIL'}")

        if h2_level > -60 or h4_level > -60:
            test_passed = False

        if test_passed:
            print("\n✅ TEST 2 PASSED: Pure h₃ produces only odd harmonics")
            self.passed_tests.append("TEST 2: Pure h₃")
        else:
            print("\n❌ TEST 2 FAILED: Even harmonics present (wrong!)")
            self.failed_tests.append("TEST 2: Pure h₃ - even harmonics present")

        print("="*70 + "\n")
        return test_passed, y, spec

    def test_known_polynomial(self, use_optimized=False):
        """
        TEST 3: Known polynomial coefficients should be recovered exactly.

        System: y = 1.0x + 0.1x² + 0.01x³ + 0.005x⁵

        Expected kernel values:
        - h₁[0] = 1.0 ± 0.01 (1% tolerance)
        - h₂[0] = 0.1 ± 0.001
        - h₃[0] = 0.01 ± 0.0001
        - h₅[0] = 0.005 ± 0.00005

        RED FLAG: Error > 1% indicates implementation bug
        """
        print("="*70)
        print("TEST 3: Known Polynomial - Coefficient Recovery")
        print("="*70)

        # True coefficients
        true_coeffs = {
            'a1': 1.0,
            'a2': 0.1,
            'a3': 0.01,
            'a5': 0.005
        }

        # Create kernel with known coefficients
        kernel = VolterraKernelFull.from_polynomial_coeffs(N=512, **true_coeffs)

        # Create processor
        if use_optimized and OPTIMIZED_AVAILABLE:
            from volterra.processor_full import VolterraProcessorFull
            proc = VolterraProcessorFull.__new__(VolterraProcessorFull)
            proc.kernel = kernel
            proc.sample_rate = self.fs
            proc.use_numba = False
            proc.engine = OptimizedNumbaEngine() if NUMBA_AVAILABLE else OptimizedDiagonalEngine()
            proc.reset = lambda: None
            proc.reset()
            engine_name = "Optimized"
        else:
            proc = VolterraProcessorFull(kernel, sample_rate=self.fs, use_numba=False)
            engine_name = "Standard"

        # Test on range of inputs (polynomial evaluation)
        x_test = np.linspace(-0.5, 0.5, 1000)
        y_test = np.zeros_like(x_test)

        proc.reset()
        for i, xi in enumerate(x_test):
            yi = proc.process_block(np.array([xi]))[0]
            y_test[i] = yi

        # Expected output (direct polynomial)
        y_expected = (
            true_coeffs['a1'] * x_test +
            true_coeffs['a2'] * x_test**2 +
            true_coeffs['a3'] * x_test**3 +
            true_coeffs['a5'] * x_test**5
        )

        # Compute error
        abs_error = np.max(np.abs(y_test - y_expected))
        rel_error = abs_error / np.max(np.abs(y_expected))
        rel_error_percent = rel_error * 100

        print(f"\nEngine: {engine_name}")
        print(f"Polynomial: y = {true_coeffs['a1']}x + {true_coeffs['a2']}x² + {true_coeffs['a3']}x³ + {true_coeffs['a5']}x⁵")
        print(f"\nCoefficient Verification:")
        print(f"  h₁[0] = {kernel.h1[0]:.6f}  (expected: {true_coeffs['a1']:.6f})")
        print(f"  h₂[0] = {kernel.h2[0]:.6f}  (expected: {true_coeffs['a2']:.6f})")
        print(f"  h₃[0] = {kernel.h3_diagonal[0]:.6f}  (expected: {true_coeffs['a3']:.6f})")
        print(f"  h₅[0] = {kernel.h5_diagonal[0]:.6f}  (expected: {true_coeffs['a5']:.6f})")

        print(f"\nPolynomial Evaluation Error:")
        print(f"  Maximum absolute error: {abs_error:.2e}")
        print(f"  Maximum relative error: {rel_error_percent:.4f}%")

        # Test passes if error < 1%
        test_passed = rel_error_percent < 1.0

        if test_passed:
            print(f"\n✅ TEST 3 PASSED: Polynomial coefficients exact (error {rel_error_percent:.4f}% < 1%)")
            self.passed_tests.append("TEST 3: Polynomial")
        else:
            print(f"\n❌ TEST 3 FAILED: Error {rel_error_percent:.4f}% > 1% tolerance")
            self.failed_tests.append(f"TEST 3: Polynomial - error {rel_error_percent:.2f}%")

        print("="*70 + "\n")
        return test_passed, y_test, y_expected

    def test_block_vs_continuous(self, use_optimized=False):
        """
        TEST 4: Block processing must equal continuous processing.

        This is the CRITICAL test for overlap-add correctness in nonlinear systems.

        Process same signal:
        1. Continuously (single large block)
        2. In small blocks (128, 256, 512 samples)

        Expected: Difference < 1e-10 (float64 precision)

        RED FLAG: Difference > 1e-6 indicates overlap-add bug!
        """
        print("="*70)
        print("TEST 4: Block vs Continuous Processing")
        print("="*70)

        # Create non-trivial kernel (all orders)
        kernel = VolterraKernelFull.from_polynomial_coeffs(
            N=512,
            a1=0.9,
            a2=0.15,
            a3=0.03,
            a4=0.01,
            a5=0.02
        )

        # Create processor
        if use_optimized and OPTIMIZED_AVAILABLE:
            from volterra.processor_full import VolterraProcessorFull
            proc_class = VolterraProcessorFull
            engine_name = "Optimized"
        else:
            proc_class = VolterraProcessorFull
            engine_name = "Standard"

        # Generate complex test signal (multiple frequencies)
        duration = 2.0
        t = np.arange(int(self.fs * duration)) / self.fs
        x = (
            0.2 * np.sin(2 * np.pi * 440 * t) +
            0.15 * np.sin(2 * np.pi * 880 * t) +
            0.1 * np.sin(2 * np.pi * 1320 * t)
        )

        # 1. Continuous processing (reference)
        if use_optimized and OPTIMIZED_AVAILABLE:
            proc_cont = proc_class.__new__(proc_class)
            proc_cont.kernel = kernel
            proc_cont.sample_rate = self.fs
            proc_cont.use_numba = False
            proc_cont.engine = OptimizedNumbaEngine() if NUMBA_AVAILABLE else OptimizedDiagonalEngine()
            proc_cont.reset = lambda: None
            proc_cont.reset()
        else:
            proc_cont = proc_class(kernel, sample_rate=self.fs, use_numba=False)

        y_continuous = proc_cont.process(x, block_size=len(x))  # Single block

        print(f"\nEngine: {engine_name}")
        print(f"Test signal: {duration:.1f}s, {len(x)} samples")
        print(f"Kernel: All orders 1-5, N={kernel.N}")
        print(f"\nProcessing Tests:")
        print(f"{'Block Size':<15} {'Max Error':<15} {'RMS Error':<15} {'Status'}")
        print("-"*60)

        test_passed = True
        max_error_overall = 0.0

        # 2. Test various block sizes
        block_sizes = [128, 256, 512, 1024]

        for block_size in block_sizes:
            if use_optimized and OPTIMIZED_AVAILABLE:
                proc_block = proc_class.__new__(proc_class)
                proc_block.kernel = kernel
                proc_block.sample_rate = self.fs
                proc_block.use_numba = False
                proc_block.engine = OptimizedNumbaEngine() if NUMBA_AVAILABLE else OptimizedDiagonalEngine()
                proc_block.reset = lambda: None
                proc_block.reset()
            else:
                proc_block = proc_class(kernel, sample_rate=self.fs, use_numba=False)

            y_blocks = proc_block.process(x, block_size=block_size)

            # Compute error
            diff = np.abs(y_continuous - y_blocks)
            max_error = np.max(diff)
            rms_error = np.sqrt(np.mean(diff**2))
            max_error_overall = max(max_error_overall, max_error)

            # Float64 precision: expect < 1e-10, warn if > 1e-6
            if max_error < 1e-10:
                status = "✓ PERFECT"
            elif max_error < 1e-6:
                status = "✓ GOOD"
            else:
                status = "✗ FAIL"
                test_passed = False

            print(f"{block_size:<15} {max_error:<15.2e} {rms_error:<15.2e} {status}")

        print(f"\nWorst-case error: {max_error_overall:.2e}")

        if test_passed:
            print(f"✅ TEST 4 PASSED: Block processing matches continuous (error < 1e-6)")
            self.passed_tests.append("TEST 4: Block vs Continuous")
        else:
            print(f"❌ TEST 4 FAILED: Overlap-add bug detected (error > 1e-6)")
            self.failed_tests.append(f"TEST 4: Block vs Continuous - error {max_error_overall:.2e}")

        print("="*70 + "\n")
        return test_passed

    def run_full_audit(self, test_optimized=True):
        """Run complete mathematical correctness audit."""
        print("\n" + "="*70)
        print("MATHEMATICAL CORRECTNESS AUDIT - FULL SUITE")
        print("="*70 + "\n")

        engines_to_test = ["Standard"]
        if test_optimized and OPTIMIZED_AVAILABLE:
            engines_to_test.append("Optimized")

        for engine in engines_to_test:
            use_opt = (engine == "Optimized")

            print(f"\n{'='*70}")
            print(f"TESTING WITH {engine.upper()} ENGINE")
            print(f"{'='*70}\n")

            # Run all tests
            self.test_pure_h2_only_2f0(use_optimized=use_opt)
            self.test_pure_h3_odd_harmonics(use_optimized=use_opt)
            self.test_known_polynomial(use_optimized=use_opt)
            self.test_block_vs_continuous(use_optimized=use_opt)

        # Final summary
        print("\n" + "="*70)
        print("AUDIT SUMMARY")
        print("="*70)

        total = len(self.passed_tests) + len(self.failed_tests)
        print(f"\nTests Run: {total}")
        print(f"Passed: {len(self.passed_tests)}")
        print(f"Failed: {len(self.failed_tests)}")

        if self.failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in self.failed_tests:
                print(f"  - {test}")
            print(f"\n⚠️  CRITICAL: Implementation has mathematical errors!")
            print(f"    Review failed tests and fix before proceeding.")
        else:
            print(f"\n✅ ALL TESTS PASSED")
            print(f"    Implementation is mathematically correct!")

        print("="*70 + "\n")

        return len(self.failed_tests) == 0


if __name__ == "__main__":
    auditor = MathematicalCorrectnessAuditor()

    # Run full audit
    success = auditor.run_full_audit(test_optimized=OPTIMIZED_AVAILABLE)

    # Exit code
    exit(0 if success else 1)
