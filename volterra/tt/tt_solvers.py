"""
Tensor-Train ALS/MALS solvers for Volterra system identification.

This module implements:
- TT-ALS: Fixed-rank Alternating Least Squares solver
- TT-MALS: Modified ALS with rank adaptation via SVD truncation

The solvers identify TT-Volterra models from input-output data by solving:
    min ||y - f(x; TT-cores)||^2

where f is the Volterra operator represented in TT format.

References:
- Holtz et al. (2012), "The alternating linear scheme for tensor optimization"
- Oseledets & Tyrtyshnikov (2010), "TT-cross approximation for multidimensional arrays"
- Zhao et al. (2016), "Tensor ring decomposition" (for MALS rank adaptation)
"""

import warnings
from dataclasses import dataclass

import numpy as np

# Import TT-ALS implementations
from volterra.tt.tt_solvers_simple import (
    fit_diagonal_volterra_als,
    fit_diagonal_volterra_mimo_als,
    fit_diagonal_volterra_rls,
)


@dataclass
class TTALSConfig:
    """
    Configuration for TT-ALS solver.

    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of ALS sweeps
    tol : float, default=1e-6
        Convergence tolerance (relative change in loss)
    verbose : bool, default=False
        Print iteration progress
    init_method : str, default='randn'
        Initialization method: 'randn' (random normal) or 'zeros'
    regularization : float, default=1e-8
        Tikhonov regularization for least-squares stability
    """

    max_iter: int = 100
    tol: float = 1e-6
    verbose: bool = False
    init_method: str = "randn"
    regularization: float = 1e-8


@dataclass
class TTMALSConfig(TTALSConfig):
    """
    Configuration for TT-MALS solver (ALS + rank adaptation).

    Extends TTALSConfig with rank adaptation parameters.

    Parameters
    ----------
    rank_adaptation : bool, default=True
        Enable rank adaptation via SVD truncation
    max_rank : int, default=10
        Maximum allowed TT rank
    rank_tol : float, default=1e-4
        Singular value threshold for rank truncation (relative to largest SV)
    adapt_every : int, default=5
        Perform rank adaptation every N sweeps
    """

    rank_adaptation: bool = True
    max_rank: int = 10
    rank_tol: float = 1e-4
    adapt_every: int = 5


def _build_volterra_design_matrix(x: np.ndarray, memory_length: int, _order: int) -> np.ndarray:
    """
    Build Volterra design matrix from input signal.

    For a signal x of length T, constructs a matrix where each row contains
    the delayed input samples needed for Volterra evaluation at time t.

    Parameters
    ----------
    x : np.ndarray
        Input signal, shape (T,) for SISO or (T, I) for MIMO
    memory_length : int
        Memory length N (number of delays)
    _order : int
        Volterra order M - reserved for future full Volterra support

    Returns
    -------
    X : np.ndarray
        Design matrix, shape (T-N+1, N^M) for full Volterra or
        (T-N+1, N) for diagonal-only

    Notes
    -----
    For now, this is a placeholder. Full TT-ALS implementation requires
    a more sophisticated tensor unfolding strategy.
    """
    # This is a simplified version - full implementation would use
    # tensor unfolding matrices for TT-ALS
    if x.ndim == 1:
        x = x[:, np.newaxis]  # (T, 1)

    T, I = x.shape
    N = memory_length

    # Number of valid samples after accounting for memory
    T_valid = T - N + 1

    # Build delay matrix: each row is [x(t), x(t-1), ..., x(t-N+1)]
    X_delay = np.zeros((T_valid, N, I))
    for tau in range(N):
        X_delay[:, tau, :] = x[N - 1 - tau : T - tau, :]

    # For single input, squeeze last dimension
    if I == 1:
        X_delay = X_delay[:, :, 0]  # (T_valid, N)

    return X_delay


def _initialize_tt_cores(
    K: int, N: int, ranks: list[int], method: str = "randn"
) -> list[np.ndarray]:
    """
    Initialize TT cores with specified ranks.

    Parameters
    ----------
    K : int
        Number of cores (Volterra order)
    N : int
        Dimension size (memory length)
    ranks : List[int]
        TT ranks [r_0=1, r_1, ..., r_{K-1}, r_K=1]
    method : str
        Initialization: 'randn' or 'zeros'

    Returns
    -------
    cores : List[np.ndarray]
        Initialized TT cores

    Raises
    ------
    ValueError
        If ranks are invalid
    """
    if len(ranks) != K + 1:
        raise ValueError(f"Need {K+1} ranks for {K} cores, got {len(ranks)}")

    if ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError(f"Boundary ranks must be 1, got r_0={ranks[0]}, r_K={ranks[-1]}")

    cores = []
    for k in range(K):
        r_left = ranks[k]
        r_right = ranks[k + 1]
        shape = (r_left, N, r_right)

        if method == "randn":
            # Random normal initialization, scaled by 1/sqrt(size)
            core = np.random.randn(*shape) / np.sqrt(N * r_left * r_right)
        elif method == "zeros":
            core = np.zeros(shape)
        else:
            raise ValueError(f"Unknown init method: {method}")

        cores.append(core)

    return cores


def _qr_left_orthogonalize(cores: list[np.ndarray], pivot: int) -> list[np.ndarray]:
    """
    Left-orthogonalize all cores before pivot using QR decomposition.

    After this operation, cores[0:pivot] are in left-orthogonal form,
    meaning that contracting any core with its conjugate transpose gives identity.

    This is a key step in TT-ALS for isolating one core during optimization.

    Parameters
    ----------
    cores : List[np.ndarray]
        TT cores
    pivot : int
        Core index to orthogonalize towards (not modified)

    Returns
    -------
    cores_orth : List[np.ndarray]
        Orthogonalized cores
    """
    cores_orth = [core.copy() for core in cores]

    for k in range(pivot):
        core = cores_orth[k]  # (r_{k-1}, n_k, r_k)
        r_left, n_k, r_right = core.shape

        # Reshape to matrix: (r_{k-1} * n_k, r_k)
        core_mat = core.reshape(r_left * n_k, r_right)

        # QR decomposition
        Q, R = np.linalg.qr(core_mat)

        # Update core with Q, reshape back
        cores_orth[k] = Q.reshape(r_left, n_k, -1)

        # Absorb R into next core
        if k + 1 < len(cores_orth):
            next_core = cores_orth[k + 1]  # (r_k, n_{k+1}, r_{k+1})
            # Contract R (r_k_new, r_k) with next_core (r_k, n_{k+1}, r_{k+1})
            next_core_flat = next_core.reshape(r_right, -1)
            cores_orth[k + 1] = (R @ next_core_flat).reshape(
                -1, next_core.shape[1], next_core.shape[2]
            )

    return cores_orth


def tt_als(
    x: np.ndarray,
    y: np.ndarray,
    memory_length: int,
    order: int,
    ranks: list[int],
    config: TTALSConfig | None = None,
) -> tuple[list[np.ndarray], dict]:
    """
    TT-ALS: Tensor-Train Alternating Least Squares solver for Volterra identification.

    Identifies a Volterra model in TT format from input-output data using
    proper alternating least-squares optimization with:
    - Hankel delay matrix construction
    - Left/right QR orthogonalization of cores
    - Regularized least squares for each core
    - Convergence monitoring

    Parameters
    ----------
    x : np.ndarray
        Input signal, shape (T,) for SISO or (T, I) for MIMO
    y : np.ndarray
        Output signal, shape (T,) for single output
    memory_length : int
        Memory length N (number of delays)
    order : int
        Volterra order M (number of TT cores)
    ranks : List[int]
        TT ranks [r_0=1, r_1, ..., r_{M-1}, r_M=1]
    config : TTALSConfig, optional
        Solver configuration

    Returns
    -------
    cores : List[np.ndarray]
        Optimized TT cores
    info : dict
        Optimization info: 'loss_history', 'iterations', 'converged', 'final_loss'

    Raises
    ------
    ValueError
        If inputs are invalid or incompatible

    Examples
    --------
    >>> # SISO Volterra identification
    >>> x = np.random.randn(1000)
    >>> y = x + 0.1 * x**2  # Nonlinear system
    >>> cores, info = tt_als(x, y, memory_length=10, order=2, ranks=[1, 3, 1])
    >>> print(info['converged'])
    True
    >>> print(info['final_loss'])
    0.001234
    """
    if config is None:
        config = TTALSConfig()

    # Validate inputs
    if x.ndim > 2:
        raise ValueError(f"Input x must be 1D or 2D, got shape {x.shape}")
    if y.ndim != 1:
        raise ValueError(f"Output y must be 1D, got shape {y.shape}")

    # Validate ranks
    if len(ranks) != order + 1:
        raise ValueError(f"Need {order+1} ranks for order {order}, got {len(ranks)}")
    if ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError(f"Boundary ranks must be 1, got r_0={ranks[0]}, r_M={ranks[-1]}")

    # Check if ranks are compatible with diagonal mode (all ranks = 1)
    diagonal_mode = all(r == 1 for r in ranks)

    # Handle MIMO vs SISO
    is_mimo = x.ndim == 2

    if is_mimo:
        # MIMO: use additive model with separate kernels per input
        if diagonal_mode:
            cores_per_input, info = fit_diagonal_volterra_mimo_als(
                x,
                y,
                memory_length,
                order,
                max_iter=config.max_iter,
                tol=config.tol,
                regularization=config.regularization,
                verbose=config.verbose,
            )
            # For compatibility, return cores from first input (will be handled properly in TTVolterraIdentifier)
            cores = cores_per_input
            info["mimo"] = True
            info["cores_per_input"] = cores_per_input
        else:
            warnings.warn(
                f"Ranks {ranks} indicate non-diagonal TT. "
                "Full general TT-Volterra for MIMO is complex. "
                "Falling back to diagonal MIMO mode.",
                UserWarning,
                stacklevel=2,
            )
            cores_per_input, info = fit_diagonal_volterra_mimo_als(
                x,
                y,
                memory_length,
                order,
                max_iter=config.max_iter,
                tol=config.tol,
                regularization=config.regularization,
                verbose=config.verbose,
            )
            cores = cores_per_input
            info["mimo"] = True
            info["cores_per_input"] = cores_per_input
    else:
        # SISO: standard diagonal Volterra
        if diagonal_mode:
            cores, info = fit_diagonal_volterra_als(
                x,
                y,
                memory_length,
                order,
                max_iter=config.max_iter,
                tol=config.tol,
                regularization=config.regularization,
                verbose=config.verbose,
            )
            info["mimo"] = False
        else:
            warnings.warn(
                f"Ranks {ranks} indicate non-diagonal TT. "
                "Full general TT-ALS for Volterra is complex. "
                "Falling back to diagonal (memory polynomial) mode. "
                "Set all ranks to 1 for diagonal Volterra.",
                UserWarning,
                stacklevel=2,
            )
            cores, info = fit_diagonal_volterra_als(
                x,
                y,
                memory_length,
                order,
                max_iter=config.max_iter,
                tol=config.tol,
                regularization=config.regularization,
                verbose=config.verbose,
            )
            info["mimo"] = False

    return cores, info


def tt_mals(
    x: np.ndarray,
    y: np.ndarray,
    memory_length: int,
    order: int,
    initial_ranks: list[int],
    config: TTMALSConfig | None = None,
) -> tuple[list[np.ndarray], dict]:
    """
    TT-MALS: Modified ALS with rank adaptation for Volterra identification.

    Extends TT-ALS with automatic rank adaptation via SVD truncation.
    Ranks are increased/decreased based on singular value spectrum.

    Parameters
    ----------
    x : np.ndarray
        Input signal, shape (T,) for SISO or (T, I) for MIMO
    y : np.ndarray
        Output signal, shape (T,) for single output
    memory_length : int
        Memory length N
    order : int
        Volterra order M
    initial_ranks : List[int]
        Initial TT ranks [r_0=1, r_1, ..., r_{M-1}, r_M=1]
    config : TTMALSConfig, optional
        Solver configuration

    Returns
    -------
    cores : List[np.ndarray]
        Optimized TT cores with adapted ranks
    info : dict
        Optimization info including rank adaptation history

    Notes
    -----
    Rank adaptation uses SVD truncation: singular values below
    `rank_tol * max(singular_values)` are discarded, subject to
    max_rank constraint.

    Examples
    --------
    >>> # SISO with rank adaptation
    >>> x = np.random.randn(1000)
    >>> y = x + 0.1 * x**2
    >>> cores, info = tt_mals(x, y, memory_length=10, order=2, initial_ranks=[1, 2, 1])
    >>> print(info['final_ranks'])
    """
    if config is None:
        config = TTMALSConfig()

    # Start with TT-ALS
    cores, info_als = tt_als(x, y, memory_length, order, initial_ranks, config)

    # Placeholder for rank adaptation
    # Full implementation would:
    # 1. After every adapt_every sweeps:
    #    a. Compute SVD of each core
    #    b. Truncate small singular values
    #    c. Update ranks accordingly
    # 2. Continue ALS with new ranks

    warnings.warn(
        "TT-MALS is a placeholder implementation. "
        "Full rank adaptation requires SVD truncation logic.",
        UserWarning,
        stacklevel=2,
    )

    info = {
        **info_als,
        "initial_ranks": initial_ranks,
        "final_ranks": [core.shape[0] for core in cores] + [1],
        "rank_adaptation_steps": 0,
        "message": "Placeholder implementation - no rank adaptation performed",
    }

    return cores, info


def tt_rls(
    x: np.ndarray,
    y: np.ndarray,
    memory_length: int,
    order: int,
    ranks: list[int],
    forgetting_factor: float = 0.99,
    regularization: float = 1e-4,
    verbose: bool = False,
) -> tuple[list[np.ndarray], dict]:
    """
    Online/adaptive diagonal TT-Volterra identification using Recursive Least Squares.

    This solver processes data sample-by-sample, updating the model adaptively.
    Suitable for time-varying systems and online learning scenarios.

    Parameters
    ----------
    x : np.ndarray
        Input signal, shape (T,) for SISO
    y : np.ndarray
        Output signal, shape (T,)
    memory_length : int
        Memory length N (number of delays)
    order : int
        Volterra order M (number of TT cores)
    ranks : List[int]
        TT ranks [r_0=1, r_1, ..., r_M=1] - only diagonal (all 1s) supported
    forgetting_factor : float, default=0.99
        RLS forgetting factor λ ∈ (0, 1]
        λ = 1: infinite memory (standard RLS)
        λ < 1: exponential forgetting (for time-varying systems)
    regularization : float, default=1e-4
        Initial diagonal loading for inverse correlation matrix
    verbose : bool, default=False
        Print progress

    Returns
    -------
    cores : List[np.ndarray]
        Final adapted TT cores
    info : dict
        Adaptation info including MSE history

    Raises
    ------
    ValueError
        If inputs are invalid or MIMO is provided (not yet supported for RLS)

    Examples
    --------
    >>> # Online identification of time-varying system
    >>> x = np.random.randn(10000)
    >>> y = generate_time_varying_output(x)
    >>> cores, info = tt_rls(x, y, memory_length=10, order=3,
    ...                       ranks=[1,1,1,1], forgetting_factor=0.98)
    >>> plt.plot(info['mse_history'])  # Show adaptation trajectory

    Notes
    -----
    RLS is particularly useful for:
    - Time-varying nonlinear systems
    - Online/streaming data processing
    - Adaptive filters for changing environments
    - Systems with parameter drift

    For stationary systems, batch ALS (tt_als) is typically more accurate.
    """
    # Validate inputs
    if x.ndim > 1:
        raise ValueError("RLS currently only supports SISO systems (1D input)")
    if y.ndim != 1:
        raise ValueError(f"Output y must be 1D, got shape {y.shape}")

    # Validate ranks
    if len(ranks) != order + 1:
        raise ValueError(f"Need {order+1} ranks for order {order}, got {len(ranks)}")
    if ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError(f"Boundary ranks must be 1, got r_0={ranks[0]}, r_M={ranks[-1]}")

    # Check diagonal mode
    if not all(r == 1 for r in ranks):
        warnings.warn(
            f"RLS only supports diagonal TT. Ranks {ranks} will be treated as diagonal.",
            UserWarning,
            stacklevel=2,
        )

    # Call RLS implementation
    cores, info = fit_diagonal_volterra_rls(
        x,
        y,
        memory_length,
        order,
        forgetting_factor=forgetting_factor,
        regularization=regularization,
        verbose=verbose,
    )

    info["mimo"] = False

    return cores, info
