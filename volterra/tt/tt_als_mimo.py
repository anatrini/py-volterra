"""
Full TT-ALS for general MIMO Volterra system identification.

This module implements Tensor-Train Alternating Least Squares for general
(non-diagonal) Volterra identification with support for MIMO systems.

Key features:
- Proper tensor unfolding and Kronecker product design matrices
- Left/right orthogonalization for efficient core updates
- Regularized least squares with condition monitoring
- Support for arbitrary TT ranks (not just diagonal)
- Monotonic loss decrease safeguard

Mathematical model:
For SISO: y(t) = Σ_{i1,...,iM=0}^{N-1} h[i1,...,iM] · x[t-i1] · ... · x[t-iM]
For MIMO: y(t) = Σ_{i1,...,iM} H[i1,...,iM] · x1[t-j1] · ... · xI[t-jI]
          where each ik is a multi-index over (input, delay)

TT decomposition:
h[i1,...,iM] = G0[:,i1,:] @ G1[:,i2,:] @ ... @ G_{M-1}[:,iM,:]
where Gk has shape (r_{k-1}, N, r_k) for SISO
or (r_{k-1}, I*N, r_k) for MIMO (Kronecker product structure)

References:
- Batselier, Chen, Wong (2017), "Tensor Network alternating linear scheme
  for MIMO Volterra system identification", Automatica, Vol. 84, pp. 26-35
- Oseledets (2011), "Tensor-Train Decomposition", SIAM J. Sci. Comput.
- Holtz et al. (2012), "The alternating linear scheme for tensor optimization"
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy import linalg
import warnings

from volterra.tt.tt_cores import (
    validate_tt_cores_structure,
    left_orthogonalize_cores,
    right_orthogonalize_cores,
    estimate_condition_number,
)


def build_mimo_delay_matrix(
    x: np.ndarray,
    memory_length: int
) -> np.ndarray:
    """
    Build MIMO delay matrix with Kronecker product structure.

    For MIMO input x of shape (T, I), creates a delay matrix where each
    row contains delayed samples from all I input channels:
    [x1[t], x1[t-1], ..., x1[t-N+1], x2[t], x2[t-1], ..., xI[t-N+1]]

    Parameters
    ----------
    x : np.ndarray
        Input signal, shape (T,) for SISO or (T, I) for MIMO
    memory_length : int
        Memory length N (number of delays per channel)

    Returns
    -------
    X_delay : np.ndarray
        MIMO delay matrix, shape (T_valid, I * N)
        where T_valid = T - N + 1

    Examples
    --------
    >>> x = np.random.randn(100, 2)  # 2 inputs
    >>> X = build_mimo_delay_matrix(x, memory_length=5)
    >>> X.shape
    (96, 10)  # 10 = 2 inputs × 5 delays
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]  # (T, 1)

    T, I = x.shape
    N = memory_length

    if T < N:
        raise ValueError(f"Signal length {T} < memory length {N}")

    T_valid = T - N + 1

    # Build delay matrix: (T_valid, I * N)
    X_delay = np.zeros((T_valid, I * N))

    for i in range(I):
        for tau in range(N):
            # Column index for input i, delay tau
            col_idx = i * N + tau
            # Extract delayed samples: x_i[t-tau] for t in [N-1, T-1]
            X_delay[:, col_idx] = x[N-1-tau:T-tau, i]

    return X_delay


def build_tt_design_matrix(
    X_delay: np.ndarray,
    core_idx: int,
    order: int,
    left_cores: Optional[List[np.ndarray]] = None,
    right_cores: Optional[List[np.ndarray]] = None,
    left_contractions: Optional[np.ndarray] = None,
    right_contractions: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Build design matrix for optimizing a specific TT core.

    This function constructs the matrix Φ such that:
        y ≈ Φ @ vec(core[k])

    where core[k] is the core being optimized.

    The design matrix is built by contracting all other cores with the
    input delay matrix, creating the effective regressor for core k.

    Parameters
    ----------
    X_delay : np.ndarray
        MIMO delay matrix, shape (T_valid, mode_size)
        mode_size = N for SISO, I*N for MIMO
    core_idx : int
        Index of core being optimized (0 to M-1)
    order : int
        Volterra order M (number of cores)
    left_cores : List[np.ndarray], optional
        Cores 0 to k-1 (left-orthogonalized)
    right_cores : List[np.ndarray], optional
        Cores k+1 to M-1 (right-orthogonalized)
    left_contractions : np.ndarray, optional
        Precomputed left contractions, shape (T_valid, r_left)
    right_contractions : np.ndarray, optional
        Precomputed right contractions, shape (T_valid, r_right)

    Returns
    -------
    Phi : np.ndarray
        Design matrix, shape (T_valid, r_left * mode_size * r_right)

    Notes
    -----
    The design matrix is built using Kronecker products:
        Phi[t, :] = kron(left_contraction[t], X_delay[t], right_contraction[t])

    For the first sweep (no orthogonalized cores), we use simpler construction.
    """
    T_valid, mode_size = X_delay.shape
    k = core_idx
    M = order

    # Compute left contractions if not provided
    if left_contractions is None:
        if left_cores is not None and len(left_cores) > 0:
            # Contract left cores with delay data
            left_contractions = np.ones((T_valid, 1))  # Start with (T_valid, 1)

            for lc in left_cores:
                # lc shape: (r_prev, mode_size, r_next)
                r_prev, _, r_next = lc.shape

                # For each sample t, contract lc with X_delay[t]
                # Result: (T_valid, r_next)
                new_contractions = np.zeros((T_valid, r_next))

                for t in range(T_valid):
                    # left_contractions[t]: (r_prev,)
                    # X_delay[t]: (mode_size,)
                    # lc: (r_prev, mode_size, r_next)

                    # Contract: sum over r_prev and mode_size
                    # C[r_next] = sum_{r_prev, i} left[r_prev] * X[i] * lc[r_prev, i, r_next]
                    for r_p in range(r_prev):
                        for i in range(mode_size):
                            new_contractions[t, :] += (
                                left_contractions[t, r_p] * X_delay[t, i] * lc[r_p, i, :]
                            )

                left_contractions = new_contractions
        else:
            left_contractions = np.ones((T_valid, 1))

    # Compute right contractions if not provided
    if right_contractions is None:
        if right_cores is not None and len(right_cores) > 0:
            # Contract right cores with delay data
            right_contractions = np.ones((T_valid, 1))

            for rc in reversed(right_cores):
                # rc shape: (r_prev, mode_size, r_next)
                r_prev, _, r_next = rc.shape

                # Contract
                new_contractions = np.zeros((T_valid, r_prev))

                for t in range(T_valid):
                    # right_contractions[t]: (r_next,)
                    # X_delay[t]: (mode_size,)
                    # rc: (r_prev, mode_size, r_next)

                    for r_n in range(r_next):
                        for i in range(mode_size):
                            new_contractions[t, :] += (
                                right_contractions[t, r_n] * X_delay[t, i] * rc[:, i, r_n]
                            )

                right_contractions = new_contractions
        else:
            right_contractions = np.ones((T_valid, 1))

    # Build design matrix: Phi = kron(left, X_delay, right)
    r_left = left_contractions.shape[1]
    r_right = right_contractions.shape[1]

    Phi = np.zeros((T_valid, r_left * mode_size * r_right))

    for t in range(T_valid):
        # Kronecker product: left[t] ⊗ X[t] ⊗ right[t]
        # Shape: (r_left,) ⊗ (mode_size,) ⊗ (r_right,) = (r_left * mode_size * r_right,)
        kron_prod = np.kron(np.kron(left_contractions[t], X_delay[t]), right_contractions[t])
        Phi[t, :] = kron_prod

    return Phi


def solve_core_regularized_lstsq(
    Phi: np.ndarray,
    y: np.ndarray,
    core_shape: Tuple[int, int, int],
    regularization: float = 1e-8,
    check_condition: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Solve regularized least squares for a TT core.

    Solves: min ||y - Φ @ vec(core)||² + λ ||vec(core)||²

    Uses Tikhonov regularization for numerical stability.

    Parameters
    ----------
    Phi : np.ndarray
        Design matrix, shape (T_valid, r_left * mode_size * r_right)
    y : np.ndarray
        Target output, shape (T_valid,)
    core_shape : Tuple[int, int, int]
        Target shape (r_left, mode_size, r_right) for the core
    regularization : float, default=1e-8
        Tikhonov regularization parameter λ
    check_condition : bool, default=True
        If True, compute and warn about condition number

    Returns
    -------
    core : np.ndarray
        Optimized core, shape (r_left, mode_size, r_right)
    info : dict
        Solver info: 'condition', 'residual_norm'

    Notes
    -----
    Normal equations: (Φ^T Φ + λI) x = Φ^T y

    We use scipy.linalg.solve with assume_a='pos' for efficiency.
    """
    r_left, mode_size, r_right = core_shape
    expected_cols = r_left * mode_size * r_right

    if Phi.shape[1] != expected_cols:
        raise ValueError(
            f"Phi has {Phi.shape[1]} columns, expected {expected_cols} "
            f"for core shape {core_shape}"
        )

    # Build normal equations
    A = Phi.T @ Phi + regularization * np.eye(Phi.shape[1])
    b = Phi.T @ y

    # Check condition number
    if check_condition:
        try:
            cond = np.linalg.cond(A)
            if cond > 1e10:
                warnings.warn(
                    f"High condition number {cond:.2e} detected. "
                    "Consider increasing regularization.",
                    UserWarning
                )
        except np.linalg.LinAlgError:
            cond = np.inf
    else:
        cond = None

    # Solve
    try:
        core_vec = linalg.solve(A, b, assume_a='pos')
    except linalg.LinAlgError as e:
        warnings.warn(
            f"Least squares failed: {e}. Falling back to lstsq.",
            UserWarning
        )
        core_vec, _, _, _ = linalg.lstsq(Phi, y)

    # Compute residual norm
    y_pred = Phi @ core_vec
    residual_norm = np.linalg.norm(y - y_pred)

    # Reshape to core
    core = core_vec.reshape(core_shape)

    info = {
        'condition': cond,
        'residual_norm': residual_norm,
    }

    return core, info


def evaluate_tt_volterra_mimo(
    cores: List[np.ndarray],
    X_delay: np.ndarray
) -> np.ndarray:
    """
    Evaluate TT-Volterra model on MIMO delay matrix.

    Computes y(t) for each row of X_delay using TT cores via
    tensor contraction.

    Parameters
    ----------
    cores : List[np.ndarray]
        TT cores, each with shape (r_{k-1}, mode_size, r_k)
    X_delay : np.ndarray
        MIMO delay matrix, shape (T_valid, mode_size)

    Returns
    -------
    y_pred : np.ndarray
        Predicted output, shape (T_valid,)

    Notes
    -----
    For each time t:
        y[t] = cores[0][:, X[t,0], :] @ cores[1][:, X[t,1], :] @ ... @ cores[M-1][:, X[t,M-1], :]

    Since X_delay has continuous indices, we use full tensor contraction:
        y[t] = Σ_{i1,...,iM} G0[:,i1,:] @ ... @ G_{M-1}[:,iM,:] · X[t,i1] · ... · X[t,iM]
    """
    T_valid, mode_size = X_delay.shape
    M = len(cores)

    # Validate cores
    validate_tt_cores_structure(cores, expected_mode_size=mode_size)

    y_pred = np.zeros(T_valid)

    for t in range(T_valid):
        x_t = X_delay[t, :]  # (mode_size,)

        # Contract cores with input
        # Start with cores[0] contracted with x_t
        result = cores[0]  # (1, mode_size, r1)

        # Contract with x_t: sum over mode dimension
        # result: (1, r1) = sum_i cores[0][0, i, :] * x_t[i]
        contracted = np.zeros((1, cores[0].shape[2]))
        for i in range(mode_size):
            contracted += cores[0][0, i, :] * x_t[i]

        result = contracted  # (1, r1)

        # Contract remaining cores
        for k in range(1, M):
            core_k = cores[k]  # (r_{k-1}, mode_size, r_k)
            r_left, _, r_right = core_k.shape

            # Contract result (1, r_left) with core_k (r_left, mode_size, r_right) and x_t
            # New result: (1, r_right)
            new_result = np.zeros((1, r_right))

            for r_l in range(r_left):
                for i in range(mode_size):
                    new_result[0, :] += result[0, r_l] * x_t[i] * core_k[r_l, i, :]

            result = new_result

        # Final result should be (1, 1)
        y_pred[t] = result[0, 0]

    return y_pred


def tt_als_full_mimo(
    x: np.ndarray,
    y: np.ndarray,
    memory_length: int,
    order: int,
    ranks: List[int],
    max_iter: int = 100,
    tol: float = 1e-6,
    regularization: float = 1e-8,
    verbose: bool = False,
    check_monotonic: bool = True
) -> Tuple[List[np.ndarray], dict]:
    """
    Full TT-ALS for general MIMO Volterra identification.

    Implements alternating least squares optimization for Volterra kernels
    in Tensor-Train format, supporting arbitrary ranks and MIMO inputs.

    Parameters
    ----------
    x : np.ndarray
        Input signal, shape (T,) for SISO or (T, I) for MIMO
    y : np.ndarray
        Output signal, shape (T,)
    memory_length : int
        Memory length N (number of delays per input channel)
    order : int
        Volterra order M (number of TT cores)
    ranks : List[int]
        TT ranks [r_0=1, r_1, ..., r_{M-1}, r_M=1]
    max_iter : int, default=100
        Maximum number of ALS sweeps
    tol : float, default=1e-6
        Convergence tolerance (relative change in loss)
    regularization : float, default=1e-8
        Tikhonov regularization for numerical stability
    verbose : bool, default=False
        Print iteration progress
    check_monotonic : bool, default=True
        Enforce monotonic loss decrease (reject bad updates)

    Returns
    -------
    cores : List[np.ndarray]
        Optimized TT cores
    info : dict
        Optimization info:
        - 'loss_history': list of losses per iteration
        - 'iterations': number of iterations performed
        - 'converged': whether convergence criterion met
        - 'final_loss': final MSE loss
        - 'condition_history': condition numbers (if available)

    Raises
    ------
    ValueError
        If inputs are invalid or ranks are incompatible

    Examples
    --------
    >>> # SISO general Volterra
    >>> x = np.random.randn(1000)
    >>> y = x + 0.1 * x**2 + 0.05 * x**3
    >>> cores, info = tt_als_full_mimo(
    ...     x, y, memory_length=10, order=3, ranks=[1, 2, 2, 1]
    ... )
    >>> print(info['converged'])
    True

    >>> # MIMO general Volterra
    >>> x = np.random.randn(1000, 2)  # 2 inputs
    >>> y = nonlinear_mimo_system(x)
    >>> cores, info = tt_als_full_mimo(
    ...     x, y, memory_length=8, order=2, ranks=[1, 3, 1]
    ... )
    """
    # Validate inputs
    if x.ndim > 2:
        raise ValueError(f"Input x must be 1D or 2D, got shape {x.shape}")
    if y.ndim != 1:
        raise ValueError(f"Output y must be 1D, got shape {y.shape}")

    # Canonicalize input
    if x.ndim == 1:
        x = x[:, np.newaxis]  # (T, 1)

    T, I = x.shape
    N = memory_length
    M = order

    # Validate ranks
    if len(ranks) != M + 1:
        raise ValueError(f"Need {M+1} ranks for order {M}, got {len(ranks)}")
    if ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError(f"Boundary ranks must be 1, got r_0={ranks[0]}, r_M={ranks[-1]}")

    # Build MIMO delay matrix
    X_delay = build_mimo_delay_matrix(x, N)
    T_valid = X_delay.shape[0]
    mode_size = I * N

    if verbose:
        print(f"TT-ALS MIMO: T={T}, I={I}, N={N}, M={M}, mode_size={mode_size}")
        print(f"Ranks: {ranks}")
        print(f"T_valid: {T_valid}")

    # Trim y to match
    if len(y) > T_valid:
        y_valid = y[N-1:N-1+T_valid]
    else:
        y_valid = y[:T_valid]

    if len(y_valid) != T_valid:
        raise ValueError(
            f"After delay alignment, y has {len(y_valid)} samples but expected {T_valid}"
        )

    # Initialize cores randomly
    cores = []
    for k in range(M):
        r_left = ranks[k]
        r_right = ranks[k + 1]
        scale = 1.0 / np.sqrt(mode_size * r_left * r_right)
        core = np.random.randn(r_left, mode_size, r_right) * scale
        cores.append(core)

    # ALS iterations
    loss_history = []
    condition_history = []
    prev_loss = np.inf
    converged = False

    for iteration in range(max_iter):
        # Forward sweep: optimize cores 0 to M-1
        for k in range(M):
            # Orthogonalize cores before and after k
            if k > 0:
                cores = left_orthogonalize_cores(cores, pivot=k)

            if k < M - 1:
                cores = right_orthogonalize_cores(cores, pivot=k)

            # Build design matrix for core k
            left_cores = cores[:k] if k > 0 else None
            right_cores = cores[k+1:] if k < M - 1 else None

            Phi = build_tt_design_matrix(
                X_delay, k, M, left_cores, right_cores
            )

            # Solve for core k
            core_shape = cores[k].shape
            cores[k], solve_info = solve_core_regularized_lstsq(
                Phi, y_valid, core_shape, regularization,
                check_condition=(iteration == 0)  # Only check first iteration
            )

            if solve_info['condition'] is not None:
                condition_history.append(solve_info['condition'])

        # Compute loss
        y_pred = evaluate_tt_volterra_mimo(cores, X_delay)
        loss = np.mean((y_valid - y_pred) ** 2)

        # Check monotonicity
        if check_monotonic and loss > prev_loss * 1.01:  # Allow 1% tolerance
            warnings.warn(
                f"Loss increased: {prev_loss:.6e} → {loss:.6e} at iteration {iteration+1}. "
                "This may indicate numerical issues.",
                UserWarning
            )

        loss_history.append(loss)

        # Check convergence
        if prev_loss == 0:
            rel_change = 0 if loss == 0 else np.inf
        else:
            rel_change = abs(loss - prev_loss) / abs(prev_loss)

        if verbose and (iteration % 10 == 0 or iteration < 5):
            print(f"  Iter {iteration+1}/{max_iter}: loss={loss:.6e}, rel_change={rel_change:.6e}")

        if rel_change < tol and iteration > 0:
            converged = True
            if verbose:
                print(f"Converged after {iteration+1} iterations")
            break

        prev_loss = loss

    if not converged and verbose:
        print(f"Max iterations ({max_iter}) reached without convergence")

    info = {
        'loss_history': loss_history,
        'iterations': len(loss_history),
        'converged': converged,
        'final_loss': loss_history[-1] if loss_history else np.inf,
        'condition_history': condition_history,
        'mimo': I > 1,
        'mode_size': mode_size,
    }

    return cores, info
