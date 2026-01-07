"""
Simplified TT-ALS implementation for Volterra systems.

This implements a working TT-ALS that properly optimizes diagonal-Volterra
(memory polynomial) models, which are the most commonly used in practice.

For full general TT-Volterra, tensor unfolding is complex. This implementation
focuses on the diagonal case which captures most practical scenarios.
"""

import numpy as np
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
        X_delay[t, :] = x[t : t + N][::-1]

    return X_delay


def evaluate_diagonal_volterra(cores: list[np.ndarray], X_delay: np.ndarray) -> np.ndarray:
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
    verbose: bool = False,
) -> tuple[list[np.ndarray], dict]:
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
    y_valid = y[N - 1 : N - 1 + T_valid]

    # Build feature matrix: [X, X^2, X^3, ..., X^M]
    # Shape: (T_valid, N*M)
    Phi_full = np.zeros((T_valid, N * M))
    for m in range(M):
        Phi_full[:, m * N : (m + 1) * N] = X_delay ** (m + 1)

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
            h_m = linalg.solve(A, b, assume_a="pos")

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
            print("Max iterations reached")

    info = {
        "loss_history": loss_history,
        "iterations": len(loss_history),
        "converged": converged,
        "final_loss": loss_history[-1] if loss_history else np.inf,
        "method": "diagonal_als",
    }

    return cores, info


def build_delay_matrix_mimo(x: np.ndarray, memory_length: int) -> list[np.ndarray]:
    """
    Build delay matrices for MIMO inputs.

    Parameters
    ----------
    x : np.ndarray
        Input signals, shape (T, I) where I is number of inputs
    memory_length : int
        Memory length N

    Returns
    -------
    X_delays : List[np.ndarray]
        List of I delay matrices, each shape (T-N+1, N)
    """
    T, I = x.shape
    X_delays = []
    for i in range(I):
        X_delays.append(build_delay_matrix_simple(x[:, i], memory_length))
    return X_delays


def fit_diagonal_volterra_mimo_als(
    x: np.ndarray,
    y: np.ndarray,
    memory_length: int,
    order: int,
    max_iter: int = 50,
    tol: float = 1e-6,
    regularization: float = 1e-8,
    verbose: bool = False,
) -> tuple[list[list[np.ndarray]], dict]:
    """
    Fit diagonal Volterra for MIMO systems using additive model.

    Model: y(t) = sum_{i=1}^I sum_{m=1}^M sum_{k=0}^{N-1} h_{i,m}[k] * x_i[t-k]^m

    Each input channel has its own set of diagonal Volterra kernels.

    Parameters
    ----------
    x : np.ndarray
        Input signals, shape (T, I)
    y : np.ndarray
        Output signal, shape (T,)
    memory_length : int
        Memory length N
    order : int
        Volterra order M
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
    cores_per_input : List[List[np.ndarray]]
        List of I core sets, each containing M cores of shape (1, N, 1)
    info : dict
        Optimization info
    """
    T, I = x.shape
    N = memory_length
    M = order

    # Build delay matrices for each input
    X_delays = build_delay_matrix_mimo(x, N)
    T_valid = X_delays[0].shape[0]

    # Align y
    y_valid = y[N - 1 : N - 1 + T_valid]

    # Initialize cores for each input channel
    cores_per_input = []
    for i in range(I):
        cores = []
        for m in range(M):
            core = np.random.randn(1, N, 1) * 0.01
            cores.append(core)
        cores_per_input.append(cores)

    loss_history = []
    prev_loss = np.inf

    for iteration in range(max_iter):
        # Optimize each input channel and each order separately
        for i in range(I):
            X_delay_i = X_delays[i]
            cores_i = cores_per_input[i]

            for m in range(M):
                # Build residual: y - contributions from all other terms
                residual = y_valid.copy()

                # Subtract contributions from all inputs and all orders except current (i, m)
                for j in range(I):
                    X_delay_j = X_delays[j]
                    cores_j = cores_per_input[j]

                    for k in range(M):
                        if j != i or k != m:
                            h_k = cores_j[k][0, :, 0]
                            X_k = X_delay_j ** (k + 1)
                            residual -= np.sum(h_k[np.newaxis, :] * X_k, axis=1)

                # Solve for h_{i,m}
                Phi_m = X_delay_i ** (m + 1)  # (T_valid, N)

                # Ridge regression
                A = Phi_m.T @ Phi_m + regularization * np.eye(N)
                b = Phi_m.T @ residual
                h_m = linalg.solve(A, b, assume_a="pos")

                # Update core
                cores_i[m][0, :, 0] = h_m

        # Compute loss
        y_pred = np.zeros(T_valid)
        for i in range(I):
            y_pred += evaluate_diagonal_volterra(cores_per_input[i], X_delays[i])

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
            print("Max iterations reached")

    info = {
        "loss_history": loss_history,
        "iterations": len(loss_history),
        "converged": converged,
        "final_loss": loss_history[-1] if loss_history else np.inf,
        "method": "mimo_diagonal_als",
        "n_inputs": I,
    }

    return cores_per_input, info


def fit_diagonal_volterra_rls(
    x: np.ndarray,
    y: np.ndarray,
    memory_length: int,
    order: int,
    forgetting_factor: float = 0.99,
    regularization: float = 1e-4,
    verbose: bool = False,
) -> tuple[list[np.ndarray], dict]:
    """
    Fit diagonal Volterra using online Recursive Least Squares (RLS).

    This implements an adaptive/online variant that updates coefficients
    sample-by-sample instead of batch processing.

    Model: y(t) = sum_{m=1}^M sum_{i=0}^{N-1} h_m[i] * x[t-i]^m

    Parameters
    ----------
    x : np.ndarray
        Input signal, shape (T,)
    y : np.ndarray
        Output signal, shape (T,)
    memory_length : int
        Memory length N
    order : int
        Volterra order M
    forgetting_factor : float, default=0.99
        RLS forgetting factor λ ∈ (0, 1]
        λ = 1: infinite memory
        λ < 1: exponential forgetting (better for time-varying systems)
    regularization : float, default=1e-4
        Initial diagonal loading for P matrix
    verbose : bool
        Print progress

    Returns
    -------
    cores : List[np.ndarray]
        Final adapted cores, each shape (1, N, 1)
    info : dict
        Adaptation info including MSE history

    Notes
    -----
    RLS updates for each sample:
    1. Prediction error: e(t) = y(t) - h^T φ(t)
    2. Kalman gain: k(t) = P(t-1) φ(t) / (λ + φ^T(t) P(t-1) φ(t))
    3. Update weights: h(t) = h(t-1) + k(t) e(t)
    4. Update covariance: P(t) = (P(t-1) - k(t) φ^T(t) P(t-1)) / λ
    """
    N = memory_length
    M = order
    T = len(x)

    # Total number of parameters
    n_params = N * M

    # Initialize weights (concatenated h_1, h_2, ..., h_M)
    h = np.zeros(n_params)

    # Initialize inverse correlation matrix P
    P = np.eye(n_params) / regularization

    # History tracking
    mse_history = []
    h_history = []

    # Process samples sequentially
    for t in range(N - 1, T):
        # Build feature vector φ(t) = [x(t)^1, x(t-1)^1, ..., x(t)^2, x(t-1)^2, ..., x(t)^M, ...]
        phi = np.zeros(n_params)
        for m in range(M):
            for k in range(N):
                if t - k >= 0:
                    phi[m * N + k] = x[t - k] ** (m + 1)

        # Prediction
        y_pred = np.dot(h, phi)

        # Prediction error
        e = y[t] - y_pred

        # Kalman gain: k = P φ / (λ + φ^T P φ)
        P_phi = P @ phi
        denom = forgetting_factor + phi @ P_phi
        k = P_phi / denom

        # Update weights: h = h + k * e
        h = h + k * e

        # Update covariance: P = (P - k φ^T P) / λ
        P = (P - np.outer(k, phi) @ P) / forgetting_factor

        # Track MSE
        mse_history.append(e**2)
        if verbose and (t - N + 1) % 100 == 0:
            avg_mse = np.mean(mse_history[-100:])
            print(f"Sample {t}: MSE={avg_mse:.6e}")

        h_history.append(h.copy())

    # Extract cores from final weights
    cores = []
    for m in range(M):
        core = np.zeros((1, N, 1))
        core[0, :, 0] = h[m * N : (m + 1) * N]
        cores.append(core)

    info = {
        "mse_history": np.array(mse_history),
        "final_mse": (
            np.mean(mse_history[-100:]) if len(mse_history) >= 100 else np.mean(mse_history)
        ),
        "h_history": np.array(h_history),
        "method": "rls",
        "forgetting_factor": forgetting_factor,
    }

    return cores, info
