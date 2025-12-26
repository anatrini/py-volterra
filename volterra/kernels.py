from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt

ArrayF = npt.NDArray[np.floating]


@dataclass(frozen=True)
class VolterraKernel2:
    """
    2nd-order Volterra kernels with enforced symmetry.
    
    Convention: y₂[n] = Σᵢⱼ h₂[i,j]·x[n-i]·x[n-j]
    """
    h1: ArrayF  # shape (N,)
    h2: ArrayF  # shape (N, N), symmetric
    
    def __post_init__(self):
        h1 = np.ascontiguousarray(self.h1, dtype=np.float64)
        h2 = np.ascontiguousarray(self.h2, dtype=np.float64)
        
        if h1.ndim != 1:
            raise ValueError("h1 must be 1D")
        if h2.ndim != 2 or h2.shape[0] != h2.shape[1]:
            raise ValueError("h2 must be square 2D")
        if h2.shape[0] != h1.shape[0]:
            raise ValueError("h1 and h2 size mismatch")
            
        # Enforce symmetry
        if not np.allclose(h2, h2.T, atol=1e-9):
            print("Warning: h2 not symmetric, enforcing symmetry")
            h2 = 0.5 * (h2 + h2.T)
            
        object.__setattr__(self, "h1", h1)
        object.__setattr__(self, "h2", h2)
    
    @property
    def N(self) -> int:
        return int(self.h1.shape[0])
    
    def to_npz(self, path: Path) -> None:
        """NPZ sufficient for Phase 1 (N=512); HDF5 for Phase 2 h3."""
        np.savez_compressed(path, h1=self.h1, h2=self.h2)
    
    @staticmethod
    def from_npz(path: Path) -> "VolterraKernel2":
        data = np.load(path)
        return VolterraKernel2(h1=data["h1"], h2=data["h2"])
    
    @staticmethod
    def from_hammerstein(h1: ArrayF, g2: ArrayF) -> "VolterraKernel2":
        """
        Build diagonal h2 (parallel Hammerstein: y₂ = g₂ ⊛ x²).
        This is what Farina swept-sine gives you directly.
        """
        h1 = np.asarray(h1, dtype=np.float64)
        g2 = np.asarray(g2, dtype=np.float64)
        if h1.shape != g2.shape:
            raise ValueError("h1 and g2 must match")
        N = h1.shape[0]
        h2 = np.zeros((N, N), dtype=np.float64)
        np.fill_diagonal(h2, g2)
        return VolterraKernel2(h1=h1, h2=h2)
    
    def low_rank_decomposition(self, energy: float = 0.999) -> Tuple[ArrayF, ArrayF]:
        """
        THE KEY OPTIMIZATION for N=512 dense h2.
        
        h₂ ≈ Σᵣ λᵣ·uᵣ·uᵣᵀ
        → y₂[n] ≈ Σᵣ λᵣ·(uᵣᵀ·xₙ)²
        
        Reduces O(N²) to O(R·N) where R << N.
        Returns: (weights: shape R, filters: shape R×N)
        """
        w, Q = np.linalg.eigh(self.h2)  # Symmetric eigendecomp
        idx = np.argsort(np.abs(w))[::-1]
        w = w[idx]
        Q = Q[:, idx]
        
        # Determine rank from energy threshold
        energy_cumsum = np.cumsum(np.abs(w))
        total = energy_cumsum[-1]
        if total > 0:
            R = int(np.searchsorted(energy_cumsum / total, energy) + 1)
        else:
            R = 0
            
        return w[:R], Q[:, :R].T  # filters is R×N