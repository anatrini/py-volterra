"""
Simplified TT-ALS implementation for Volterra systems.

This implements a working TT-ALS that properly optimizes diagonal-Volterra
(memory polynomial) models, which are the most commonly used in practice.

For full general TT-Volterra, tensor unfolding is complex. This implementation
focuses on the diagonal case which captures most practical scenarios.
"""

import numpy as np
from typing import List, Tuple
from scipy import linalg


def build_delay_matrix_simple(x: np.ndarray, memory_length: int) -> np.ndarray:
    """
    Build delay matrix from input signal.

    Row t contains: [x(t+N-1), x(t+N-2), ..., x(t)]

    Parameters
    ----------
    x : np.ndarray
        Input signal, shape (T,)
    memory_length : int
        Memory length N

    Returns
    -------
    X_delay : np.ndarray
        Delay matrix, shape (T-N+1, N)
    """
    T = len(x)
    N = memory_length
    T_valid = T - N + 1

    X_delay = np.zeros((T_valid, N))
    for t in range(T_valid):
        # Most recent to oldest: x[t+N-1], ..., x[t]
        X_delay[t, :] = x[t:t+N][::-1]

    return X_delay


def evaluate_diagonal_volterra(cores: List[np.ndarray], X_delay: np.ndarray) -> np.ndarray:
    """
    Evaluate diagonal Volterra model.

    For diagonal Volterra:  y(t) = sum_{i} h1[i]*x[t-i] + sum_{i} h2[i]*x[t-i]^2 + ...

    In TT format with rank-1 diagonal cores:
    G_k has shape (1, N, 1), contains h_k[i]

    y(t) = G0[i] * x[t-i] + G1[i] * x[t-i]^2 + ... for i=0..N-1

    Parameters
    ----------
    cores : List[np.ndarray]
        TT cores, each shape (1, N, 1) for diagonal
    X_delay : np.ndarray
        Delay matrix, shape (T_valid, N)

    Returns
    -------
    y_pred : np.ndarray
        Predictions, shape (T_valid,)
    """
    T_valid, N = X_delay.shape
    M = len(cores)

    y_pred = np.zeros(T_valid)

    for m in range(M):
        # Extract diagonal kernel for order m+1
        h_m = cores[m][0, :, 0]  # Shape (N,)

        # Compute contribution: sum_i h_m[i] * x[t-i]^(m+1)
        X_power = X_delay ** (m + 1)  # Element-wise power
        y_pred += np.sum(h_m[np.newaxis, :] * X_power, axis=1)

    return y_pred


def fit_diagonal_volterra_als(
    x: np.ndarray,
    y: np.ndarray,
    memory_length: int,
    order: int,
    max_iter: int = 50,
    tol: float = 1e-6,
    regularization: float = 1e-8,
    verbose: bool = False
) -> Tuple[List[np.ndarray], dict]:
    """
    Fit diagonal Volterra (memory polynomial) using alternating least squares.

    Model: y(t) = sum_{m=1}^M sum_{i=0}^{N-1} h_m[i] * x[t-i]^m

    This is a rank-1 TT decomposition where each core is (1, N, 1).

    Parameters
    ----------
    x : np.ndarray
        Input signal, shape (T,)
    y : np.ndarray
        Output signal, shape (T,)
    memory_length : int
        Memory length N
    order : int
        Volterra order M (polynomial order)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    regularization : float
        Ridge regression parameter
    verbose : bool
        Print progress

    Returns
    -------
    cores : List[np.ndarray]
        Optimized cores, each shape (1, N, 1)
    info : dict
        Optimization info
    """
    N = memory_length
    M = order

    # Build delay matrix
    X_delay = build_delay_matrix_simple(x, N)
    T_valid = X_delay.shape[0]

    # Align y
    y_valid = y[N-1:N-1+T_valid]

    # Build feature matrix: [X, X^2, X^3, ..., X^M]
    # Shape: (T_valid, N*M)
    Phi_full = np.zeros((T_valid, N * M))
    for m in range(M):
        Phi_full[:, m*N:(m+1)*N] = X_delay ** (m + 1)

    # Initialize cores
    cores = []
    for m in range(M):
        core = np.random.randn(1, N, 1) * 0.01
        cores.append(core)

    loss_history = []
    prev_loss = np.inf

    for iteration in range(max_iter):
        # Optimize each core separately (coordinate descent)
        for m in range(M):
            # Build residual: y - sum_{k != m} contribution_k
            residual = y_valid.copy()
            for k in range(M):
                if k != m:
                    h_k = cores[k][0, :, 0]
                    X_k = X_delay ** (k + 1)
                    residual -= np.sum(h_k[np.newaxis, :] * X_k, axis=1)

            # Solve for h_m: residual ≈ sum_i h_m[i] * x[t-i]^(m+1)
            Phi_m = X_delay ** (m + 1)  # (T_valid, N)

            # Ridge regression
            A = Phi_m.T @ Phi_m + regularization * np.eye(N)
            b = Phi_m.T @ residual
            h_m = linalg.solve(A, b, assume_a='pos')

            # Update core
            cores[m][0, :, 0] = h_m

        # Compute loss
        y_pred = evaluate_diagonal_volterra(cores, X_delay)
        loss = np.mean((y_valid - y_pred) ** 2)
        loss_history.append(loss)

        # Check convergence
        rel_change = abs(loss - prev_loss) / (abs(prev_loss) + 1e-12)

        if verbose and iteration % 10 == 0:
            print(f"Iter {iteration}: loss={loss:.6e}, rel_change={rel_change:.6e}")

        if rel_change < tol:
            if verbose:
                print(f"Converged after {iteration+1} iterations")
            converged = True
            break

        prev_loss = loss
    else:
        converged = False
        if verbose:
            print(f"Max iterations reached")

    info = {
        'loss_history': loss_history,
        'iterations': len(loss_history),
        'converged': converged,
        'final_loss': loss_history[-1] if loss_history else np.inf,
        'method': 'diagonal_als'
    }

    return cores, info
