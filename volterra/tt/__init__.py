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

from volterra.tt.tt_tensor import (
    TTTensor,
    validate_tt_cores,
    tt_matvec,
    tt_to_full,
)

from volterra.tt.tt_solvers import (
    tt_als,
    tt_mals,
    tt_rls,
    TTALSConfig,
    TTMALSConfig,
)

from volterra.tt.tt_cores import (
    validate_tt_cores_structure,
    left_orthogonalize_cores,
    right_orthogonalize_cores,
    truncate_core_svd,
    merge_two_cores,
    split_core_svd,
    estimate_condition_number,
)

from volterra.tt.tt_als_mimo import (
    build_mimo_delay_matrix,
    build_tt_design_matrix,
    solve_core_regularized_lstsq,
    evaluate_tt_volterra_mimo,
    tt_als_full_mimo,
)

__all__ = [
    # TT tensor basics
    "TTTensor",
    "validate_tt_cores",
    "tt_matvec",
    "tt_to_full",
    # Solvers (high-level)
    "tt_als",
    "tt_mals",
    "tt_rls",
    "TTALSConfig",
    "TTMALSConfig",
    # Core utilities
    "validate_tt_cores_structure",
    "left_orthogonalize_cores",
    "right_orthogonalize_cores",
    "truncate_core_svd",
    "merge_two_cores",
    "split_core_svd",
    "estimate_condition_number",
    # Full TT-ALS for MIMO
    "build_mimo_delay_matrix",
    "build_tt_design_matrix",
    "solve_core_regularized_lstsq",
    "evaluate_tt_volterra_mimo",
    "tt_als_full_mimo",
]
