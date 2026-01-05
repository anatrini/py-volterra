"""
Shape validation and canonicalization for MIMO Volterra systems.

MIMO Data Conventions
---------------------
The library uses the following conventions for input/output data:

Input x:
    - SISO/SIMO: shape (T,) - single time series
    - MISO/MIMO: shape (T, I) - I input channels

Output y:
    - SISO/MISO: shape (T,) - single output
    - SIMO/MIMO: shape (T, O) - O output channels

Internally, all processing works with canonicalized shapes:
    - x_canon: (T, I) where I=1 for SISO
    - y_canon: (T, O) where O=1 for SISO

This module provides utilities to:
1. Canonicalize inputs/outputs to (T, I) / (T, O) format
2. Validate shape compatibility
3. Infer dimensions T, I, O from data
"""

import numpy as np


def canonicalize_input(x: np.ndarray) -> np.ndarray:
    """
    Canonicalize input data to shape (T, I).

    Converts 1D input (T,) to 2D (T, 1) for uniform MIMO processing.

    Parameters
    ----------
    x : np.ndarray
        Input data, shape (T,) for SISO or (T, I) for MIMO

    Returns
    -------
    x_canon : np.ndarray
        Canonicalized input, shape (T, I)

    Raises
    ------
    ValueError
        If input has invalid shape (not 1D or 2D)

    Examples
    --------
    >>> x_siso = np.random.randn(1000)
    >>> x_canon = canonicalize_input(x_siso)
    >>> x_canon.shape
    (1000, 1)

    >>> x_mimo = np.random.randn(1000, 3)
    >>> x_canon = canonicalize_input(x_mimo)
    >>> x_canon.shape
    (1000, 3)
    """
    if x.ndim == 1:
        # SISO: (T,) -> (T, 1)
        return x[:, np.newaxis]
    elif x.ndim == 2:
        # Already MIMO format
        return x
    else:
        raise ValueError(f"Input must be 1D (T,) or 2D (T, I), got shape {x.shape}")


def canonicalize_output(y: np.ndarray) -> np.ndarray:
    """
    Canonicalize output data to shape (T, O).

    Converts 1D output (T,) to 2D (T, 1) for uniform MIMO processing.

    Parameters
    ----------
    y : np.ndarray
        Output data, shape (T,) for single output or (T, O) for MIMO

    Returns
    -------
    y_canon : np.ndarray
        Canonicalized output, shape (T, O)

    Raises
    ------
    ValueError
        If output has invalid shape (not 1D or 2D)

    Examples
    --------
    >>> y_siso = np.random.randn(1000)
    >>> y_canon = canonicalize_output(y_siso)
    >>> y_canon.shape
    (1000, 1)

    >>> y_mimo = np.random.randn(1000, 2)
    >>> y_canon = canonicalize_output(y_mimo)
    >>> y_canon.shape
    (1000, 2)
    """
    if y.ndim == 1:
        # Single output: (T,) -> (T, 1)
        return y[:, np.newaxis]
    elif y.ndim == 2:
        # Already multi-output format
        return y
    else:
        raise ValueError(f"Output must be 1D (T,) or 2D (T, O), got shape {y.shape}")


def validate_mimo_data(x: np.ndarray, y: np.ndarray, require_same_length: bool = True) -> None:
    """
    Validate that input and output data have compatible shapes.

    Checks:
    1. Both x and y are 1D or 2D arrays
    2. First dimension (time T) matches if require_same_length=True
    3. Arrays are not empty

    Parameters
    ----------
    x : np.ndarray
        Input data, shape (T,) or (T, I)
    y : np.ndarray
        Output data, shape (T,) or (T, O)
    require_same_length : bool, default=True
        If True, enforce x.shape[0] == y.shape[0]

    Raises
    ------
    ValueError
        If shapes are incompatible or data is invalid

    Examples
    --------
    >>> x = np.random.randn(1000, 2)
    >>> y = np.random.randn(1000, 3)
    >>> validate_mimo_data(x, y)  # OK

    >>> y_wrong = np.random.randn(500, 3)
    >>> validate_mimo_data(x, y_wrong)  # Raises ValueError
    """
    # Check x is valid
    if x.ndim not in (1, 2):
        raise ValueError(f"Input x must be 1D (T,) or 2D (T, I), got shape {x.shape}")

    # Check y is valid
    if y.ndim not in (1, 2):
        raise ValueError(f"Output y must be 1D (T,) or 2D (T, O), got shape {y.shape}")

    # Check not empty
    if x.shape[0] == 0:
        raise ValueError("Input x cannot be empty (T=0)")
    if y.shape[0] == 0:
        raise ValueError("Output y cannot be empty (T=0)")

    # Check time dimension matches
    if require_same_length and x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Input and output must have same length, got x.shape={x.shape}, " f"y.shape={y.shape}"
        )


def infer_dimensions(x: np.ndarray, y: np.ndarray) -> tuple[int, int, int]:
    """
    Infer dimensions T, I, O from input/output data.

    Parameters
    ----------
    x : np.ndarray
        Input data, shape (T,) or (T, I)
    y : np.ndarray
        Output data, shape (T,) or (T, O)

    Returns
    -------
    T : int
        Number of time samples
    I : int
        Number of input channels (1 for SISO)
    O : int
        Number of output channels (1 for SISO)

    Raises
    ------
    ValueError
        If shapes are invalid or incompatible

    Examples
    --------
    >>> x = np.random.randn(1000, 3)
    >>> y = np.random.randn(1000, 2)
    >>> T, I, O = infer_dimensions(x, y)
    >>> T, I, O
    (1000, 3, 2)

    >>> x_siso = np.random.randn(500)
    >>> y_siso = np.random.randn(500)
    >>> T, I, O = infer_dimensions(x_siso, y_siso)
    >>> T, I, O
    (500, 1, 1)
    """
    # Validate first
    validate_mimo_data(x, y)

    # Extract dimensions
    T = x.shape[0]
    I = 1 if x.ndim == 1 else x.shape[1]
    O = 1 if y.ndim == 1 else y.shape[1]

    return T, I, O
