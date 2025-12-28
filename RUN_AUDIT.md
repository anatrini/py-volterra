# How to Run Mathematical Correctness Audit

## Quick Start

```bash
# Run complete audit
uv run python examples/mathematical_correctness_audit.py
```

## What This Tests

### ✅ Test 1: Pure h₂ → Only 2f₀
Input sine wave should produce ONLY the 2nd harmonic.
- **Pass:** 2f₀ strong, f₀ and 3f₀ suppressed
- **Fail:** If f₀ or 3f₀ appear → wrong implementation

### ✅ Test 2: Pure h₃ → Only Odd Harmonics  
Input sine should produce ONLY odd harmonics (no even).
- **Pass:** 2f₀ and 4f₀ suppressed (< -60 dB)
- **Fail:** If even harmonics appear → wrong implementation

### ✅ Test 3: Known Polynomial → Exact Coefficients
System y = a₁x + a₂x² + a₃x³ + a₅x⁵ should evaluate exactly.
- **Pass:** Error < 1%
- **Fail:** Error > 1% → bug in kernel evaluation

### ✅ Test 4: Block vs Continuous → Identical Output
Same signal processed as one block vs many blocks should match.
- **Pass:** Difference < 1e-10
- **Fail:** Difference > 1e-6 → overlap-add bug

## Expected Output

```
======================================================================
MATHEMATICAL CORRECTNESS AUDIT - FULL SUITE
======================================================================

TESTING WITH STANDARD ENGINE
======================================================================

TEST 1: Pure h₂ - Should produce ONLY 2f₀
...
✅ TEST 1 PASSED: Pure h₂ produces only 2f₀ as expected

TEST 2: Pure h₃ - Should produce ONLY odd harmonics
...
✅ TEST 2 PASSED: Pure h₃ produces only odd harmonics

TEST 3: Known Polynomial - Coefficient Recovery
...
✅ TEST 3 PASSED: Polynomial coefficients exact (error 0.0001% < 1%)

TEST 4: Block vs Continuous Processing
...
✅ TEST 4 PASSED: Block processing matches continuous (error < 1e-6)

======================================================================
AUDIT SUMMARY
======================================================================

Tests Run: 4
Passed: 4
Failed: 0

✅ ALL TESTS PASSED
    Implementation is mathematically correct!
```

## If Tests Fail

### Red Flag: Test 1 Fails (Pure h₂)
**Symptom:** Fundamental (f₀) or 3rd harmonic (3f₀) appear
**Cause:** Powers computed incorrectly or wrong kernel
**Action:** Check `_numba_convolve_diagonal_order2` - should compute `x*x`

### Red Flag: Test 2 Fails (Pure h₃)
**Symptom:** Even harmonics (2f₀, 4f₀) present
**Cause:** Cross-contamination between orders or wrong cubic computation
**Action:** Verify no h₂ or h₄ terms active; check `x*x*x` computation

### Red Flag: Test 3 Fails (Polynomial)
**Symptom:** Error > 1% when evaluating polynomial
**Cause:** Kernel coefficients wrong or evaluation bug
**Action:** Check coefficient extraction and summation logic

### Red Flag: Test 4 Fails (Block vs Continuous)
**Symptom:** Different output for block vs continuous processing
**Cause:** Overlap-add bug - history buffer mismanaged
**Action:** Check history concatenation and power computation on extended signal

## Testing Optimized Engine

The audit automatically tests both standard and optimized engines if available.

Optimized engine features:
- Power computation using reuse chain (x² → x³ → x⁴ → x⁵)
- 2-3x faster than standard implementation
- Mathematically identical results

## Interpreting Results

### All Pass → Implementation Correct ✅
- Safe to use for audio processing
- Mathematical correctness verified
- Can proceed with confidence

### Any Fail → Critical Bug ❌
- DO NOT use for production
- Review implementation against audit report
- Fix issues before proceeding

## Performance Notes

- Test suite runs ~8 tests (4 × 2 engines)
- Takes ~10-30 seconds depending on machine
- Uses float64 for maximum precision
- Generates detailed harmonic analysis

## Next Steps After Passing

1. Review `MATHEMATICAL_AUDIT_REPORT.md` for optimization opportunities
2. Consider using optimized engine for production (3x faster)
3. Run performance benchmarks for your specific use case
4. Integrate into your audio pipeline

## Technical Details

**Precision:**
- float64 throughout for maximum accuracy
- Tolerance: < 1e-10 for block matching
- Polynomial error: < 1% for coefficient recovery

**Test Signal Parameters:**
- Sample rate: 48000 Hz
- Test frequency: 1000 Hz
- Amplitude: 0.2-0.3 (moderate)
- Duration: 1-2 seconds

**Kernel Parameters:**
- Length N = 512 samples
- Block sizes tested: 128, 256, 512, 1024
- Orders: 1, 2, 3, 4, 5
