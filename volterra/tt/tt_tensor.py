"""
Tensor-Train (TT) tensor representation and operations.

The TT format represents a K-dimensional tensor as a product of K cores:
    A[i_0, i_1, ..., i_{K-1}] = G_0[i_0] @ G_1[i_1] @ ... @ G_{K-1}[i_{K-1}]

Each core G_k has shape (r_{k-1}, n_k, r_k), where:
- n_k is the dimension size for index k
- r_k is the TT rank between cores k and k+1
- r_0 = r_K = 1 (boundary conditions)

For Volterra systems with memory N and order M, we have:
- K = M (M cores, one per order)
- n_k = N for all k (uniform memory length)
- r_k are adaptive ranks determined during identification

References:
- Oseledets (2011), "Tensor-Train Decomposition", SIAM J. Sci. Comput.
- Bigoni et al. (2016), "Spectral TT cross approximation of parametric functions"
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class TTTensor:
    """
    Tensor-Train tensor representation.

    Attributes
    ----------
    cores : List[np.ndarray]
        List of K TT cores, where cores[k] has shape (r_{k-1}, n_k, r_k)
    """

    cores: list[np.ndarray]

    def __post_init__(self):
        """Validate TT cores on construction."""
        validate_tt_cores(self.cores)

    @property
    def ndim(self) -> int:
        """Number of dimensions (K)."""
        return len(self.cores)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the full tensor (n_0, n_1, ..., n_{K-1})."""
        return tuple(core.shape[1] for core in self.cores)

    @property
    def ranks(self) -> tuple[int, ...]:
        """TT ranks (r_0, r_1, ..., r_K)."""
        if len(self.cores) == 0:
            return ()
        ranks = [self.cores[0].shape[0]]  # r_0
        for core in self.cores:
            ranks.append(core.shape[2])  # r_k
        return tuple(ranks)

    def __repr__(self) -> str:
        """String representation showing shape and ranks."""
        return f"TTTensor(shape={self.shape}, ranks={self.ranks})"


def validate_tt_cores(cores: list[np.ndarray]) -> None:
    """
    Validate that TT cores have compatible shapes.

    Checks:
    1. All cores are 3D arrays
    2. Boundary conditions: r_0 = 1 and r_K = 1
    3. Rank compatibility: cores[k].shape[2] == cores[k+1].shape[0]
    4. All cores are non-empty

    Parameters
    ----------
    cores : List[np.ndarray]
        List of K TT cores

    Raises
    ------
    ValueError
        If cores are invalid or incompatible

    Examples
    --------
    >>> # Valid TT cores for 3D tensor (4 x 5 x 6) with ranks (1, 2, 3, 1)
    >>> cores = [
    ...     np.random.randn(1, 4, 2),  # (r_0=1, n_0=4, r_1=2)
    ...     np.random.randn(2, 5, 3),  # (r_1=2, n_1=5, r_2=3)
    ...     np.random.randn(3, 6, 1),  # (r_2=3, n_2=6, r_3=1)
    ... ]
    >>> validate_tt_cores(cores)  # OK
    """
    if len(cores) == 0:
        raise ValueError("TT cores list cannot be empty")

    for k, core in enumerate(cores):
        # Check dimensionality
        if core.ndim != 3:
            raise ValueError(f"Core {k} must be 3D array, got shape {core.shape}")

        # Check not empty
        if core.size == 0:
            raise ValueError(f"Core {k} is empty")

        r_left, n_k, r_right = core.shape

        # Check boundary conditions
        if k == 0 and r_left != 1:
            raise ValueError(f"First core must have r_0=1 (left rank), got {r_left}")
        if k == len(cores) - 1 and r_right != 1:
            raise ValueError(f"Last core must have r_K=1 (right rank), got {r_right}")

        # Check rank compatibility with next core
        if k < len(cores) - 1:
            next_r_left = cores[k + 1].shape[0]
            if r_right != next_r_left:
                raise ValueError(
                    f"Rank mismatch between cores {k} and {k+1}: "
                    f"core[{k}].shape[2]={r_right}, "
                    f"core[{k+1}].shape[0]={next_r_left}"
                )


def tt_matvec(cores: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    """
    Tensor-Train matrix-vector multiplication for Volterra evaluation.

    For a TT tensor representing a Volterra kernel, computes the output
    given an input signal x with memory N.

    This implements the contraction:
        y = sum_{i_0,...,i_{K-1}} A[i_0,...,i_{K-1}] * x[i_0] * ... * x[i_{K-1}]

    where A is represented in TT format by the cores.

    For Volterra systems:
    - x represents delayed input samples: x = [x(n), x(n-1), ..., x(n-N+1)]
    - K is the Volterra order
    - Output is a scalar contribution to y(n)

    Parameters
    ----------
    cores : List[np.ndarray]
        TT cores, cores[k] has shape (r_{k-1}, n_k, r_k)
    x : np.ndarray
        Input vector, shape (N,) where N matches core dimension sizes

    Returns
    -------
    y : float
        Scalar output from TT-vector contraction

    Raises
    ------
    ValueError
        If x shape doesn't match TT core dimensions

    Examples
    --------
    >>> # 2nd-order Volterra kernel, N=3, rank=2
    >>> cores = [
    ...     np.random.randn(1, 3, 2),  # h1 contribution
    ...     np.random.randn(2, 3, 1),  # h2 contribution
    ... ]
    >>> x = np.array([1.0, 0.5, 0.1])  # input memory buffer
    >>> y = tt_matvec(cores, x)
    >>> isinstance(y, (float, np.floating))
    True
    """
    validate_tt_cores(cores)

    len(cores)
    N = x.shape[0]

    # Check dimension compatibility
    for k, core in enumerate(cores):
        n_k = core.shape[1]
        if n_k != N:
            raise ValueError(f"Core {k} has dimension n_{k}={n_k}, but input has length {N}")

    # Start with scalar 1.0 (r_0 = 1)
    result = np.array([[1.0]])  # Shape (1, r_0=1)

    # Contract each core with input vector
    for k, core in enumerate(cores):
        r_left, n_k, r_right = core.shape

        # Contract core with input: sum_i core[:, i, :] * x[i]
        # Result shape: (r_left, r_right)
        contracted = np.tensordot(core, x, axes=([1], [0]))  # (r_left, r_right)

        # Multiply with accumulated result: result @ contracted
        # result shape: (1, r_left) @ (r_left, r_right) = (1, r_right)
        result = result @ contracted

    # Final result is (1, 1), extract scalar
    return float(result[0, 0])


def tt_to_full(cores: list[np.ndarray]) -> np.ndarray:
    """
    Materialize full tensor from TT cores.

    WARNING: This creates an N^K array, which can be extremely large!
    Only use for debugging/validation with small N and K.

    For a TT with K cores and dimensions (n_0, n_1, ..., n_{K-1}),
    constructs the full tensor of shape (n_0, n_1, ..., n_{K-1}).

    Parameters
    ----------
    cores : List[np.ndarray]
        TT cores

    Returns
    -------
    A : np.ndarray
        Full tensor, shape (n_0, n_1, ..., n_{K-1})

    Raises
    ------
    ValueError
        If cores are invalid

    Examples
    --------
    >>> # Small example: 2D tensor (3 x 4) with rank 2
    >>> cores = [
    ...     np.ones((1, 3, 2)),
    ...     np.ones((2, 4, 1)),
    ... ]
    >>> A = tt_to_full(cores)
    >>> A.shape
    (3, 4)
    """
    validate_tt_cores(cores)

    # Start with first core, remove r_0=1 dimension
    A = cores[0][0, :, :]  # Shape (n_0, r_1)

    # Contract remaining cores
    for k in range(1, len(cores)):
        core_k = cores[k]  # Shape (r_k, n_k, r_{k+1})
        r_left, n_k, r_right = core_k.shape

        # Reshape A to (..., r_k)
        A_shape = A.shape
        A_flat = A.reshape(-1, r_left)  # Shape (prod(n_0...n_{k-1}), r_k)

        # Contract: A_flat @ core_k.reshape(r_k, n_k * r_{k+1})
        core_flat = core_k.reshape(r_left, n_k * r_right)
        A_flat = A_flat @ core_flat  # Shape (prod(n_0...n_{k-1}), n_k * r_{k+1})

        # Reshape to (..., n_k, r_{k+1})
        A = A_flat.reshape(*A_shape[:-1], n_k, r_right)

    # Remove last r_K=1 dimension
    A = A[..., 0]

    return A
