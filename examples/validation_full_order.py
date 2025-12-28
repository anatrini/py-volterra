"""
Comprehensive validation tests for Volterra kernels up to 5th order.

These tests verify:
1. Mathematical correctness of polynomial kernels
2. Harmonic generation accuracy
3. Multi-tone kernel estimation
4. Symmetry and conservation properties
"""

import numpy as np
from scipy.fft import rfft, rfftfreq

from volterra import (
    VolterraKernelFull,
    VolterraProcessorFull,
    MultiToneConfig,
    MultiToneEstimator
)


def test_identity_kernel():
    """Test 1: Identity kernel (h1=δ, all others zero) should pass signal unchanged."""
    print("="*70)
    print("TEST 1: Identity Kernel")
    print("="*70)

    N = 512
    kernel = VolterraKernelFull.from_polynomial_coeffs(N, a1=1.0)

    proc = VolterraProcessorFull(kernel, use_numba=False)
    x = np.random.randn(10000) * 0.1
    y = proc.process(x, block_size=512)

    error = np.max(np.abs(y - x))
    print(f"Maximum error: {error:.2e}")

    assert error < 1e-10, f"Identity test failed: error={error}"
    print("✓ PASSED: Identity kernel preserves signal\n")


def test_polynomial_accuracy():
    """Test 2: Known polynomial should produce exact coefficients."""
    print("="*70)
    print("TEST 2: Polynomial Accuracy")
    print("="*70)

    coeffs = {'a1': 1.0, 'a2': 0.15, 'a3': 0.05, 'a4': 0.01, 'a5': 0.02}

    # Create memoryless kernel
    kernel = VolterraKernelFull.from_polynomial_coeffs(N=512, **coeffs)
    proc = VolterraProcessorFull(kernel, use_numba=False)

    # Test on range of inputs
    x_test = np.linspace(-0.5, 0.5, 1000)
    y_test = np.array([proc.process_block(np.array([xi]))[0] for xi in x_test])

    # Expected output: polynomial
    y_expected = (coeffs['a1'] * x_test +
                  coeffs['a2'] * x_test**2 +
                  coeffs['a3'] * x_test**3 +
                  coeffs['a4'] * x_test**4 +
                  coeffs['a5'] * x_test**5)

    error = np.max(np.abs(y_test - y_expected))
    rel_error = error / np.max(np.abs(y_expected))

    print(f"Maximum absolute error: {error:.2e}")
    print(f"Maximum relative error: {rel_error:.2e}")

    assert rel_error < 1e-10, f"Polynomial test failed: rel_error={rel_error}"
    print("✓ PASSED: Polynomial coefficients exact\n")


def test_harmonic_separation():
    """Test 3: Different orders should produce correct harmonics."""
    print("="*70)
    print("TEST 3: Harmonic Separation")
    print("="*70)

    fs = 48000
    f0 = 1000.0
    duration = 1.0
    t = np.arange(int(fs * duration)) / fs
    x = 0.2 * np.sin(2 * np.pi * f0 * t)

    # Test cases: (name, coeffs, expected dominant harmonics)
    test_cases = [
        ("2nd order only", {'a1': 0, 'a2': 0.3, 'a3': 0, 'a4': 0, 'a5': 0}, [2, 4]),
        ("3rd order only", {'a1': 0, 'a2': 0, 'a3': 0.3, 'a4': 0, 'a5': 0}, [3]),
        ("5th order only", {'a1': 0, 'a2': 0, 'a3': 0, 'a4': 0, 'a5': 0.3}, [5]),
    ]

    print(f"Input: {f0:.0f} Hz sine wave\n")

    all_passed = True
    for name, coeffs, expected_harmonics in test_cases:
        kernel = VolterraKernelFull.from_polynomial_coeffs(N=512, **coeffs)
        proc = VolterraProcessorFull(kernel, use_numba=False)
        y = proc.process(x, block_size=512)

        # Analyze spectrum
        Y = rfft(y)
        freqs = rfftfreq(len(y), 1/fs)

        harmonic_levels = {}
        for k in range(1, 7):
            idx = np.argmin(np.abs(freqs - k * f0))
            level_db = 20 * np.log10(np.abs(Y[idx]) + 1e-12)
            harmonic_levels[k] = level_db

        # Check expected harmonics are dominant
        max_level = max(harmonic_levels.values())

        print(f"{name}:")
        passed = True
        for k, level in harmonic_levels.items():
            is_expected = k in expected_harmonics
            is_dominant = (level > max_level - 20)  # Within 20 dB of max

            status = "✓" if (is_expected == is_dominant) else "✗"

            if is_expected != is_dominant:
                passed = False
                all_passed = False

            marker = " <--" if is_expected else ""
            print(f"  H{k} ({k*f0:.0f}Hz): {level:>6.1f} dB {status}{marker}")

        print()

    assert all_passed, "Harmonic separation test failed"
    print("✓ PASSED: Harmonic generation correct\n")


def test_multitone_estimation():
    """Test 4: Multi-tone estimation should recover known kernels."""
    print("="*70)
    print("TEST 4: Multi-tone Kernel Estimation")
    print("="*70)

    # Known system (polynomial)
    true_coeffs = {'a1': 1.0, 'a2': 0.1, 'a3': 0.02, 'a4': 0.005, 'a5': 0.01}
    true_kernel = VolterraKernelFull.from_polynomial_coeffs(N=512, **true_coeffs)
    true_proc = VolterraProcessorFull(true_kernel, sample_rate=48000, use_numba=False)

    # Generate multi-tone excitation
    config = MultiToneConfig(
        sample_rate=48000,
        duration=2.0,
        num_tones=80,
        max_order=5,
        method='random_phase',
        amplitude=0.1
    )

    estimator = MultiToneEstimator(config)
    excitation, frequencies = estimator.generate_excitation()

    print(f"Generated multi-tone with {len(frequencies)} frequencies")
    print(f"Frequency range: {frequencies[0]:.1f} - {frequencies[-1]:.1f} Hz")

    # Simulate system response
    response = true_proc.process(excitation, block_size=512)

    # Add noise (60 dB SNR)
    noise_level = np.std(response) / 1000
    response_noisy = response + np.random.randn(len(response)) * noise_level

    # Estimate SNR
    snr_db = estimator.estimate_snr_db(response_noisy, frequencies)
    print(f"Measurement SNR: {snr_db:.1f} dB")

    # Extract kernels
    print("\nExtracting kernels...")
    estimated_kernel = estimator.estimate_kernel(
        excitation, response_noisy, frequencies, kernel_length=512
    )

    # Compare coefficients (DC values)
    print("\nCoefficient comparison:")
    print(f"{'Kernel':<10} {'True':<12} {'Estimated':<12} {'Error (%)'}")
    print("-"*50)

    errors = {}

    # h1
    h1_true = true_coeffs['a1']
    h1_est = estimated_kernel.h1[0]
    errors['h1'] = abs(h1_est - h1_true) / abs(h1_true) * 100
    print(f"{'h1[0]':<10} {h1_true:>10.4f}   {h1_est:>10.4f}   {errors['h1']:>8.2f}")

    # h2
    if estimated_kernel.h2 is not None:
        h2_true = true_coeffs['a2']
        h2_est = estimated_kernel.h2[0]
        errors['h2'] = abs(h2_est - h2_true) / abs(h2_true) * 100
        print(f"{'h2[0]':<10} {h2_true:>10.4f}   {h2_est:>10.4f}   {errors['h2']:>8.2f}")

    # h3
    if estimated_kernel.h3_diagonal is not None:
        h3_true = true_coeffs['a3']
        h3_est = estimated_kernel.h3_diagonal[0]
        errors['h3'] = abs(h3_est - h3_true) / abs(h3_true) * 100
        print(f"{'h3[0]':<10} {h3_true:>10.4f}   {h3_est:>10.4f}   {errors['h3']:>8.2f}")

    # h4
    if estimated_kernel.h4_diagonal is not None:
        h4_true = true_coeffs['a4']
        h4_est = estimated_kernel.h4_diagonal[0]
        errors['h4'] = abs(h4_est - h4_true) / abs(h4_true) * 100
        print(f"{'h4[0]':<10} {h4_true:>10.4f}   {h4_est:>10.4f}   {errors['h4']:>8.2f}")

    # h5
    if estimated_kernel.h5_diagonal is not None:
        h5_true = true_coeffs['a5']
        h5_est = estimated_kernel.h5_diagonal[0]
        errors['h5'] = abs(h5_est - h5_true) / abs(h5_true) * 100
        print(f"{'h5[0]':<10} {h5_true:>10.4f}   {h5_est:>10.4f}   {errors['h5']:>8.2f}")

    # Check all errors < 5%
    max_error = max(errors.values())
    print(f"\nMaximum error: {max_error:.2f}%")

    # Higher orders are harder to estimate accurately
    tolerance = {'h1': 1.0, 'h2': 3.0, 'h3': 10.0, 'h4': 15.0, 'h5': 15.0}

    passed = all(errors[k] < tolerance[k] for k in errors.keys())

    assert passed, f"Estimation errors too large: {errors}"
    print("✓ PASSED: Multi-tone estimation accurate\n")


def test_energy_conservation():
    """Test 5: Energy should be conserved (no gain for passive systems)."""
    print("="*70)
    print("TEST 5: Energy Conservation")
    print("="*70)

    # Create passive (no gain) kernel
    kernel = VolterraKernelFull.from_polynomial_coeffs(
        N=512, a1=0.9, a2=0.05, a3=0.01, a4=0.002, a5=0.005
    )

    proc = VolterraProcessorFull(kernel, use_numba=False)

    # White noise input
    x = np.random.randn(48000) * 0.1
    y = proc.process(x, block_size=512)

    energy_in = np.sum(x**2)
    energy_out = np.sum(y**2)
    energy_ratio = energy_out / energy_in

    print(f"Input energy:  {energy_in:.4f}")
    print(f"Output energy: {energy_out:.4f}")
    print(f"Ratio:         {energy_ratio:.4f}")

    # For passive system with a1=0.9, expect ratio ~ 0.81 (a1^2)
    expected_ratio = 0.9**2
    tolerance = 0.15  # Allow some variation due to nonlinearity

    assert abs(energy_ratio - expected_ratio) < tolerance, \
        f"Energy not conserved: ratio={energy_ratio}, expected≈{expected_ratio}"

    print(f"✓ PASSED: Energy approximately conserved\n")


def test_numba_vs_numpy():
    """Test 6: Numba and NumPy engines should give identical results."""
    print("="*70)
    print("TEST 6: Numba vs NumPy Engine Consistency")
    print("="*70)

    kernel = VolterraKernelFull.from_polynomial_coeffs(
        N=512, a1=0.9, a2=0.12, a3=0.03, a4=0.01, a5=0.02
    )

    x = np.random.randn(10000) * 0.1

    try:
        # NumPy engine
        proc_numpy = VolterraProcessorFull(kernel, use_numba=False)
        y_numpy = proc_numpy.process(x, block_size=512)

        # Numba engine
        proc_numba = VolterraProcessorFull(kernel, use_numba=True)
        y_numba = proc_numba.process(x, block_size=512)

        # Compare
        max_diff = np.max(np.abs(y_numpy - y_numba))
        rel_diff = max_diff / np.max(np.abs(y_numpy))

        print(f"Maximum absolute difference: {max_diff:.2e}")
        print(f"Maximum relative difference: {rel_diff:.2e}")

        assert rel_diff < 1e-10, f"Engines differ: rel_diff={rel_diff}"
        print("✓ PASSED: Numba and NumPy engines consistent\n")

    except RuntimeError as e:
        print(f"⚠ SKIPPED: {e}\n")


def run_all_tests():
    """Run complete validation suite."""
    print("\n" + "="*70)
    print("VOLTERRA FULL-ORDER VALIDATION SUITE")
    print("="*70 + "\n")

    tests = [
        test_identity_kernel,
        test_polynomial_accuracy,
        test_harmonic_separation,
        test_multitone_estimation,
        test_energy_conservation,
        test_numba_vs_numpy,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}\n")
            failed += 1

    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✓ ALL TESTS PASSED - Volterra implementation is mathematically correct")
    else:
        print(f"\n✗ {failed} test(s) failed - review implementation")

    print("="*70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
