"""
Tensor-Train (TT) decomposition primitives for Volterra system identification.

This module provides low-level TT decomposition infrastructure:
- TT tensor representation and storage
- TT-matrix-vector multiplication
- TT-ALS / TT-MALS solvers for system identification
- Core utilities (orthogonalization, rank truncation)
- Full TT-ALS for general MIMO Volterra

The TT format avoids the curse of dimensionality for high-order Volterra kernels
by representing N^M tensors using O(M * N * r^2) parameters, where r is the TT rank.

Internal module - not exposed in public API.
"""

from volterra.tt.tt_als_mimo import (
    build_mimo_delay_matrix,
    build_tt_design_matrix,
    evaluate_tt_volterra_mimo,
    solve_core_regularized_lstsq,
    tt_als_full_mimo,
)
from volterra.tt.tt_cores import (
    estimate_condition_number,
    left_orthogonalize_cores,
    merge_two_cores,
    right_orthogonalize_cores,
    split_core_svd,
    truncate_core_svd,
    validate_tt_cores_structure,
)
from volterra.tt.tt_solvers import (
    TTALSConfig,
    TTMALSConfig,
    tt_als,
    tt_mals,
    tt_rls,
)
from volterra.tt.tt_tensor import (
    TTTensor,
    tt_matvec,
    tt_to_full,
    validate_tt_cores,
)

__all__ = [
    "TTALSConfig",
    "TTMALSConfig",
    # TT tensor basics
    "TTTensor",
    # Full TT-ALS for MIMO
    "build_mimo_delay_matrix",
    "build_tt_design_matrix",
    "estimate_condition_number",
    "evaluate_tt_volterra_mimo",
    "left_orthogonalize_cores",
    "merge_two_cores",
    "right_orthogonalize_cores",
    "solve_core_regularized_lstsq",
    "split_core_svd",
    "truncate_core_svd",
    # Solvers (high-level)
    "tt_als",
    "tt_als_full_mimo",
    "tt_mals",
    "tt_matvec",
    "tt_rls",
    "tt_to_full",
    "validate_tt_cores",
    # Core utilities
    "validate_tt_cores_structure",
]
