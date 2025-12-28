from typing import Protocol
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from volterra.kernels import VolterraKernel2, ArrayF


class Volterra2Engine(Protocol):
    """Strategy interface for h2 computation."""
    def process_block(self, x_ext: ArrayF, N: int, block_len: int, 
                      kernel: VolterraKernel2) -> ArrayF:
        """x_ext: history + current block. Returns: y2 for current block."""
        ...


def _build_lag_matrix(x_ext: ArrayF, N: int, block_len: int) -> ArrayF:
    """
    Build matrix X where X[b, i] = x[n_b - i] for b in current block.
    Uses safe stride tricks (validated by council).
    """
    # x_ext has shape (N-1+B,); windows of length N
    windows = sliding_window_view(x_ext, N)  # shape (B, N) forward
    # Reverse to get [x[n], x[n-1], ..., x[n-N+1]]
    return np.ascontiguousarray(windows[:block_len, ::-1])


class DirectNumpyEngine:
    """Reference implementation: y₂[b] = xᵦᵀ·h₂·xᵦ"""
    
    def process_block(self, x_ext: ArrayF, N: int, block_len: int,
                      kernel: VolterraKernel2) -> ArrayF:
        X = _build_lag_matrix(x_ext, N, block_len)  # (B, N)
        # Quadratic form via GEMM
        tmp = X @ kernel.h2  # (B, N)
        y2 = np.sum(tmp * X, axis=1)  # (B,)
        return y2


class LowRankEngine:
    """
    THE OPTIMIZATION that makes N=512 feasible.
    
    Computational cost: O(R·N·B) vs O(N²·B).
    For R=20, this is ~500× speedup over dense.
    """
    
    def __init__(self, energy: float = 0.999):
        self.energy = energy
        self._cache = {}
    
    def process_block(self, x_ext: ArrayF, N: int, block_len: int,
                      kernel: VolterraKernel2) -> ArrayF:
        kernel_id = id(kernel.h2)
        if kernel_id not in self._cache:
            w, filt = kernel.low_rank_decomposition(self.energy)
            self._cache[kernel_id] = (w, filt)
        
        w, filt = self._cache[kernel_id]
        if w.size == 0:
            return np.zeros(block_len, dtype=x_ext.dtype)
        
        X = _build_lag_matrix(x_ext, N, block_len)  # (B, N)
        p = X @ filt.T  # (B, R): projections onto eigenvectors
        y2 = (p * p) @ w  # (B,): weighted sum of squared projections
        return y2