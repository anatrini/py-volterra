"""
Block vs Continuous Processing Tests.

CRITICAL: Block processing MUST equal continuous processing.
If these tests fail → overlap-add bug (WRONG IMPLEMENTATION).

Tolerance: <1e-10 (float64) or <1e-6 (float32)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from volterra import VolterraKernelFull, VolterraProcessorFull


class TestBlockProcessing:
    """Verify block processing equals continuous processing."""

    @pytest.fixture
    def sample_rate(self):
        return 48000

    @pytest.fixture
    def test_audio_signal(self, sample_rate):
        """Complex test signal: multiple frequencies + noise."""
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Multiple sine waves
        audio = (
            0.25 * np.sin(2 * np.pi * 440 * t)
            + 0.20 * np.sin(2 * np.pi * 880 * t)
            + 0.15 * np.sin(2 * np.pi * 1320 * t)
            + 0.05 * np.random.randn(len(t))
        )

        return audio.astype(np.float64)

    @pytest.fixture
    def realistic_kernel(self):
        """Realistic kernel with all orders."""
        return VolterraKernelFull.from_polynomial_coeffs(
            N=512, a1=0.9, a2=0.15, a3=0.03, a4=0.01, a5=0.02
        )

    @pytest.mark.parametrize("block_size", [64, 128, 256, 512, 1024, 2048])
    def test_block_equals_continuous(
        self, realistic_kernel, test_audio_signal, sample_rate, block_size
    ):
        """
        Block processing must equal continuous processing.

        This is THE critical test for overlap-add correctness.
        """
        audio = test_audio_signal

        # 1. Continuous processing (single large block)
        proc_continuous = VolterraProcessorFull(
            realistic_kernel, sample_rate=sample_rate, use_numba=False
        )
        output_continuous = proc_continuous.process(audio, block_size=len(audio))

        # 2. Block processing
        proc_blocks = VolterraProcessorFull(
            realistic_kernel, sample_rate=sample_rate, use_numba=False
        )
        output_blocks = proc_blocks.process(audio, block_size=block_size)

        # 3. Verify equivalence
        max_diff = np.max(np.abs(output_continuous - output_blocks))
        rms_diff = np.sqrt(np.mean((output_continuous - output_blocks) ** 2))

        # Float64 precision: expect < 1e-10
        assert (
            max_diff < 1e-10
        ), f"Block size {block_size}: max difference {max_diff:.2e} > 1e-10 (overlap-add bug!)"
        assert rms_diff < 1e-11, f"Block size {block_size}: RMS difference {rms_diff:.2e} > 1e-11"

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_block_processing_different_dtypes(
        self, realistic_kernel, test_audio_signal, sample_rate, dtype
    ):
        """Test block processing with different dtypes."""
        audio = test_audio_signal.astype(dtype)
        block_size = 512

        proc_continuous = VolterraProcessorFull(
            realistic_kernel, sample_rate=sample_rate, use_numba=False
        )
        proc_blocks = VolterraProcessorFull(
            realistic_kernel, sample_rate=sample_rate, use_numba=False
        )

        output_continuous = proc_continuous.process(audio, block_size=len(audio))
        output_blocks = proc_blocks.process(audio, block_size=block_size)

        # Tolerance depends on dtype
        if dtype == np.float32:
            rtol, atol = 1e-6, 1e-8
        else:  # float64
            rtol, atol = 1e-10, 1e-12

        assert_allclose(
            output_continuous, output_blocks, rtol=rtol, atol=atol, err_msg=f"dtype={dtype}"
        )

    def test_reset_state_functionality(self, realistic_kernel, test_audio_signal, sample_rate):
        """Verify reset_state() clears history buffer correctly."""
        processor = VolterraProcessorFull(
            realistic_kernel, sample_rate=sample_rate, use_numba=False
        )

        # Process some audio
        block1 = test_audio_signal[:1000]
        processor.process_block(block1)

        # Reset state
        processor.reset()

        # Process again - should be as if starting fresh
        processor2 = VolterraProcessorFull(
            realistic_kernel, sample_rate=sample_rate, use_numba=False
        )
        out2_fresh = processor2.process_block(block1)

        out2_reset = processor.process_block(block1)

        # Outputs should match
        assert_allclose(
            out2_reset,
            out2_fresh,
            rtol=1e-12,
            atol=1e-14,
            err_msg="reset() didn't clear state properly",
        )

    def test_consecutive_blocks_continuity(self, realistic_kernel, test_audio_signal, sample_rate):
        """
        Processing consecutive blocks should maintain continuity.

        This tests that history buffer is correctly passed between blocks.
        """
        audio = test_audio_signal
        block_size = 256

        # Process in blocks
        processor = VolterraProcessorFull(
            realistic_kernel, sample_rate=sample_rate, use_numba=False
        )

        output_blocks = []
        for i in range(0, len(audio), block_size):
            block = audio[i : i + block_size]
            out_block = processor.process_block(block)
            output_blocks.append(out_block)

        output_blocked = np.concatenate(output_blocks)

        # Process continuously
        processor_cont = VolterraProcessorFull(
            realistic_kernel, sample_rate=sample_rate, use_numba=False
        )
        output_continuous = processor_cont.process(audio, block_size=len(audio))

        # Should be identical
        assert_allclose(
            output_blocked,
            output_continuous,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Block continuity broken",
        )

    def test_variable_block_sizes(self, realistic_kernel, test_audio_signal, sample_rate):
        """Test with variable block sizes (realistic streaming scenario)."""
        audio = test_audio_signal

        # Process with variable block sizes
        processor = VolterraProcessorFull(
            realistic_kernel, sample_rate=sample_rate, use_numba=False
        )

        block_sizes = [128, 256, 512, 256, 128, 1024, 64]
        output_blocks = []
        pos = 0

        for block_size in block_sizes:
            if pos >= len(audio):
                break

            block = audio[pos : pos + block_size]
            out_block = processor.process_block(block)
            output_blocks.append(out_block)
            pos += block_size

        # Process remaining
        if pos < len(audio):
            block = audio[pos:]
            out_block = processor.process_block(block)
            output_blocks.append(out_block)

        output_variable = np.concatenate(output_blocks)

        # Process continuously
        processor_cont = VolterraProcessorFull(
            realistic_kernel, sample_rate=sample_rate, use_numba=False
        )
        output_continuous = processor_cont.process(
            audio[: len(output_variable)], block_size=len(audio)
        )

        # Should match
        assert_allclose(
            output_variable,
            output_continuous[: len(output_variable)],
            rtol=1e-10,
            atol=1e-12,
            err_msg="Variable block sizes failed",
        )

    def test_partial_block_processing(self, realistic_kernel, sample_rate):
        """Test processing of partial blocks (last block smaller than block_size)."""
        # Audio length not divisible by block_size
        audio = np.random.randn(1337) * 0.2

        processor = VolterraProcessorFull(
            realistic_kernel, sample_rate=sample_rate, use_numba=False
        )

        # Process in 512-sample blocks (last block will be 1337 % 512 = 313 samples)
        output = processor.process(audio, block_size=512)

        # Should have same length as input
        assert len(output) == len(audio), "Output length mismatch"

        # Verify against continuous
        processor_cont = VolterraProcessorFull(
            realistic_kernel, sample_rate=sample_rate, use_numba=False
        )
        output_cont = processor_cont.process(audio, block_size=len(audio))

        assert_allclose(
            output, output_cont, rtol=1e-10, atol=1e-12, err_msg="Partial block processing failed"
        )

    def test_single_sample_blocks(self, realistic_kernel, sample_rate):
        """Extreme test: process one sample at a time."""
        audio = np.random.randn(100) * 0.2

        processor = VolterraProcessorFull(
            realistic_kernel, sample_rate=sample_rate, use_numba=False
        )

        # Process sample by sample
        output_samples = []
        for sample in audio:
            out = processor.process_block(np.array([sample]))
            output_samples.append(out[0])

        output_single = np.array(output_samples)

        # Verify against continuous
        processor_cont = VolterraProcessorFull(
            realistic_kernel, sample_rate=sample_rate, use_numba=False
        )
        output_cont = processor_cont.process(audio, block_size=len(audio))

        assert_allclose(
            output_single,
            output_cont,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Single-sample processing failed",
        )

    def test_very_long_signal(self, realistic_kernel, sample_rate):
        """Test on very long signal (10 seconds) to catch accumulation errors."""
        duration = 10.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.2 * np.sin(2 * np.pi * 440 * t)

        # Block processing
        processor_blocks = VolterraProcessorFull(
            realistic_kernel, sample_rate=sample_rate, use_numba=False
        )
        output_blocks = processor_blocks.process(audio, block_size=512)

        # Continuous processing
        processor_cont = VolterraProcessorFull(
            realistic_kernel, sample_rate=sample_rate, use_numba=False
        )
        output_cont = processor_cont.process(audio, block_size=len(audio))

        # No accumulation of errors
        assert_allclose(
            output_blocks,
            output_cont,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Long signal accumulation error",
        )


class TestStateManagement:
    """Test internal state management of processor."""

    def test_history_buffer_length(self):
        """Verify history buffer has correct length (N-1)."""
        kernel = VolterraKernelFull.from_polynomial_coeffs(N=512, a1=1.0, a2=0.1)
        processor = VolterraProcessorFull(kernel, sample_rate=48000, use_numba=False)

        # Check initial state
        assert len(processor._x_history) == 511, "History buffer should be N-1 samples"
        assert np.all(processor._x_history == 0), "History should be initialized to zero"

    def test_history_updates_after_block(self):
        """Verify history buffer updates after processing each block."""
        kernel = VolterraKernelFull.from_polynomial_coeffs(N=512, a1=1.0)
        processor = VolterraProcessorFull(kernel, sample_rate=48000, use_numba=False)

        # Process a block
        block = np.arange(256, dtype=np.float64) / 256.0
        _ = processor.process_block(block)

        # History should contain last N-1 samples of extended signal
        # Extended signal = [zeros(511), block]
        # After processing, history = last 511 samples = block[-255:] + 256 zeros from before
        # Actually: history = x_ext[-(N-1):] where x_ext = [old_history, block]

        # The last 256 samples of the block should be in history
        expected_in_history = block[-256:]
        actual_in_history = processor._x_history[-256:]

        assert_allclose(
            actual_in_history,
            expected_in_history,
            rtol=1e-12,
            atol=1e-14,
            err_msg="History not updated correctly",
        )
