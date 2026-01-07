"""
Pipeline compositions for complex audio/acoustic systems.

This module provides composable building blocks for realistic system modeling:
- Nonlinear + Linear chains (e.g., instrument → room)
- Multi-stage processing pipelines
- MIMO routing and mixing
"""

from volterra.pipelines.acoustic_chain import (
    AcousticChainConfig,
    NonlinearThenRIR,
)

__all__ = [
    "AcousticChainConfig",
    "NonlinearThenRIR",
]
