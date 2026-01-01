"""
Tensor-Train (TT) decomposition primitives for Volterra system identification.

This module provides low-level TT decomposition infrastructure:
- TT tensor representation and storage
- TT-matrix-vector multiplication
- TT-ALS / TT-MALS solvers for system identification

The TT format avoids the curse of dimensionality for high-order Volterra kernels
by representing N^M tensors using O(M * N * r^2) parameters, where r is the TT rank.

Internal module - not exposed in public API.
"""

from volterra.tt.tt_tensor import (
    TTTensor,
    validate_tt_cores,
    tt_matvec,
    tt_to_full,
)

from volterra.tt.tt_solvers import (
    tt_als,
    tt_mals,
    TTALSConfig,
    TTMALSConfig,
)

__all__ = [
    "TTTensor",
    "validate_tt_cores",
    "tt_matvec",
    "tt_to_full",
    "tt_als",
    "tt_mals",
    "TTALSConfig",
    "TTMALSConfig",
]
