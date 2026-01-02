"""
Sliding-window prediction for TT-Volterra models.

Implements proper causal filtering with memory for Volterra prediction.
"""

import numpy as np
from typing import List
from volterra.tt.tt_tensor import tt_matvec
from volterra.tt.tt_solvers_simple import (
    evaluate_diagonal_volterra,
    build_delay_matrix_simple,
    build_delay_matrix_mimo,
)


def predict_diagonal_volterra(
    cores: List[np.ndarray],
    x: np.ndarray,
    memory_length: int
) -> np.ndarray:
    """
    Predict using diagonal Volterra (memory polynomial) model.

    Uses sliding window to properly handle causality and memory.

    Parameters
    ----------
    cores : List[np.ndarray]
        TT cores for diagonal Volterra, each shape (1, N, 1)
    x : np.ndarray
        Input signal, shape (T,)
    memory_length : int
        Memory length N

    Returns
    -------
    y_pred : np.ndarray
        Predicted output, shape (T - N + 1,)

    Notes
    -----
    For memoryless (N=1), output length equals input length.
    For memory N > 1, output starts at sample N-1 (after first full window).
    """
    # Build delay matrix
    X_delay = build_delay_matrix_simple(x, memory_length)

    # Evaluate using diagonal implementation
    y_pred = evaluate_diagonal_volterra(cores, X_delay)

    return y_pred


def predict_general_volterra_sliding(
    cores: List[np.ndarray],
    x: np.ndarray,
    memory_length: int
) -> np.ndarray:
    """
    Predict using general TT-Volterra with sliding window.

    Evaluates Volterra model sample-by-sample using TT-matvec.

    Parameters
    ----------
    cores : List[np.ndarray]
        TT cores
    x : np.ndarray
        Input signal, shape (T,)
    memory_length : int
        Memory length N

    Returns
    -------
    y_pred : np.ndarray
        Predicted output, shape (T - N + 1,)

    Notes
    -----
    This uses TT-matvec for each time sample, which is general but slower
    than the diagonal-optimized version.
    """
    T = len(x)
    N = memory_length
    T_valid = T - N + 1

    if T_valid <= 0:
        raise ValueError(f"Signal length {T} too short for memory {N}")

    y_pred = np.zeros(T_valid)

    # Sliding window prediction
    for t in range(T_valid):
        # Extract memory window: x[t+N-1], x[t+N-2], ..., x[t]
        # This is the most recent N samples up to time t+N-1
        x_window = x[t:t+N][::-1]  # Reverse to get [newest, ..., oldest]

        # Evaluate using TT-matvec
        y_pred[t] = tt_matvec(cores, x_window)

    return y_pred


def predict_with_warmup(
    cores: List[np.ndarray],
    x: np.ndarray,
    memory_length: int,
    warmup_value: float = 0.0,
    diagonal_mode: bool = True
) -> np.ndarray:
    """
    Predict with warmup padding to match input length.

    Pads initial samples with warmup_value to produce output of same length as input.

    Parameters
    ----------
    cores : List[np.ndarray]
        TT cores
    x : np.ndarray
        Input signal, shape (T,)
    memory_length : int
        Memory length N
    warmup_value : float
        Value to use for initial (N-1) samples
    diagonal_mode : bool
        Whether to use diagonal-optimized evaluation

    Returns
    -------
    y_pred : np.ndarray
        Predicted output, shape (T,) - same as input

    Notes
    -----
    First (N-1) samples are set to warmup_value since the model has no prior history.
    From sample N-1 onwards, predictions use actual input history.
    """
    T = len(x)
    N = memory_length

    # Predict on available data
    if diagonal_mode:
        y_valid = predict_diagonal_volterra(cores, x, N)
    else:
        y_valid = predict_general_volterra_sliding(cores, x, N)

    # Pad with warmup
    y_full = np.full(T, warmup_value)
    y_full[N-1:] = y_valid

    return y_full


def predict_diagonal_volterra_mimo(
    cores_per_input: List[List[np.ndarray]],
    x: np.ndarray,
    memory_length: int
) -> np.ndarray:
    """
    Predict using diagonal Volterra MIMO model (additive).

    Model: y(t) = sum_{i=1}^I sum_{m=1}^M sum_{k=0}^{N-1} h_{i,m}[k] * x_i[t-k]^m

    Parameters
    ----------
    cores_per_input : List[List[np.ndarray]]
        List of I core sets, each containing M cores of shape (1, N, 1)
    x : np.ndarray
        Input signals, shape (T, I)
    memory_length : int
        Memory length N

    Returns
    -------
    y_pred : np.ndarray
        Predicted output, shape (T - N + 1,)

    Notes
    -----
    Additive MIMO model: each input has its own Volterra kernels,
    outputs are summed.
    """
    T, I = x.shape

    # Build delay matrices for each input
    X_delays = build_delay_matrix_mimo(x, memory_length)
    T_valid = X_delays[0].shape[0]

    # Initialize output
    y_pred = np.zeros(T_valid)

    # Sum contributions from all inputs
    for i in range(I):
        y_pred += evaluate_diagonal_volterra(cores_per_input[i], X_delays[i])

    return y_pred
