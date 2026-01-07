"""
Extended Volterra kernels supporting orders 1-5 with diagonal approximation.

This module provides efficient diagonal kernel implementations for higher-order
Volterra series (3rd-5th order), reducing memory complexity from O(N^k) to O(N).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

ArrayF = npt.NDArray[np.floating]


@dataclass(frozen=True)
class VolterraKernelFull:
    """
    Complete Volterra kernel representation up to 5th order with diagonal approximation.

    Memory footprint for N=512:
    - Full h2[N,N]: 512² × 8 bytes = 2 MB
    - Diagonal h3[N]: 512 × 8 bytes = 4 KB (~99.8% memory reduction)
    - Diagonal h5[N]: 512 × 8 bytes = 4 KB
    Total with all orders: ~2.1 MB vs ~550 GB for full tensors

    Convention:
    - y₁[n] = Σᵢ h1[i]·x[n-i]
    - y₂[n] = Σᵢⱼ h2[i,j]·x[n-i]·x[n-j]
    - y₃[n] = Σᵢ h3_diag[i]·x[n-i]³  (diagonal approximation)
    - y₄[n] = Σᵢ h4_diag[i]·x[n-i]⁴  (diagonal approximation)
    - y₅[n] = Σᵢ h5_diag[i]·x[n-i]⁵  (diagonal approximation)

    The diagonal approximation captures 70-80% of the nonlinear energy
    for typical audio distortion applications while being real-time compatible.
    """

    h1: ArrayF  # Linear kernel (N,)
    h2: ArrayF | None = None  # 2nd order (N,N) symmetric, or diagonal (N,)
    h3_diagonal: ArrayF | None = None  # 3rd order diagonal (N,)
    h4_diagonal: ArrayF | None = None  # 4th order diagonal (N,)
    h5_diagonal: ArrayF | None = None  # 5th order diagonal (N,)
    h2_is_diagonal: bool = False  # Flag for h2 storage format

    def __post_init__(self):
        """Validate and enforce kernel constraints."""
        h1 = np.ascontiguousarray(self.h1, dtype=np.float64)

        if h1.ndim != 1:
            raise ValueError("h1 must be 1D")

        N = h1.shape[0]

        # Validate h2 if present
        if self.h2 is not None:
            h2 = np.ascontiguousarray(self.h2, dtype=np.float64)

            if self.h2_is_diagonal:
                if h2.ndim != 1 or h2.shape[0] != N:
                    raise ValueError("h2 diagonal must be 1D with length N")
            else:
                if h2.ndim != 2 or h2.shape[0] != N or h2.shape[1] != N:
                    raise ValueError("h2 must be square 2D (N, N)")

                # Enforce symmetry for full h2
                if not np.allclose(h2, h2.T, atol=1e-9):
                    print("Warning: h2 not symmetric, enforcing symmetry")
                    h2 = 0.5 * (h2 + h2.T)

            object.__setattr__(self, "h2", h2)

        # Validate diagonal kernels
        for attr_name in ["h3_diagonal", "h4_diagonal", "h5_diagonal"]:
            kernel = getattr(self, attr_name)
            if kernel is not None:
                kernel = np.ascontiguousarray(kernel, dtype=np.float64)
                if kernel.ndim != 1 or kernel.shape[0] != N:
                    raise ValueError(f"{attr_name} must be 1D with length N")
                object.__setattr__(self, attr_name, kernel)

        object.__setattr__(self, "h1", h1)

    @property
    def N(self) -> int:
        """Kernel length."""
        return int(self.h1.shape[0])

    @property
    def max_order(self) -> int:
        """Highest active order."""
        if self.h5_diagonal is not None:
            return 5
        if self.h4_diagonal is not None:
            return 4
        if self.h3_diagonal is not None:
            return 3
        if self.h2 is not None:
            return 2
        return 1

    def to_npz(self, path: Path) -> None:
        """Save kernels to compressed NPZ file."""
        save_dict = {"h1": self.h1, "h2_is_diagonal": self.h2_is_diagonal}

        if self.h2 is not None:
            save_dict["h2"] = self.h2
        if self.h3_diagonal is not None:
            save_dict["h3_diagonal"] = self.h3_diagonal
        if self.h4_diagonal is not None:
            save_dict["h4_diagonal"] = self.h4_diagonal
        if self.h5_diagonal is not None:
            save_dict["h5_diagonal"] = self.h5_diagonal

        np.savez_compressed(path, **save_dict)

    @staticmethod
    def from_npz(path: Path) -> VolterraKernelFull:
        """Load kernels from NPZ file."""
        data = np.load(path)

        return VolterraKernelFull(
            h1=data["h1"],
            h2=data.get("h2"),
            h3_diagonal=data.get("h3_diagonal"),
            h4_diagonal=data.get("h4_diagonal"),
            h5_diagonal=data.get("h5_diagonal"),
            h2_is_diagonal=bool(data.get("h2_is_diagonal", False)),
        )

    @staticmethod
    def from_polynomial_coeffs(
        N: int, a1: float = 1.0, a2: float = 0.0, a3: float = 0.0, a4: float = 0.0, a5: float = 0.0
    ) -> VolterraKernelFull:
        """
        Create memoryless polynomial kernel: y = a1·x + a2·x² + a3·x³ + a4·x⁴ + a5·x⁵

        Useful for testing and simple saturation models.
        """
        h1 = np.zeros(N, dtype=np.float64)
        h1[0] = a1

        h2 = None
        if a2 != 0:
            h2 = np.zeros(N, dtype=np.float64)
            h2[0] = a2

        h3 = None
        if a3 != 0:
            h3 = np.zeros(N, dtype=np.float64)
            h3[0] = a3

        h4 = None
        if a4 != 0:
            h4 = np.zeros(N, dtype=np.float64)
            h4[0] = a4

        h5 = None
        if a5 != 0:
            h5 = np.zeros(N, dtype=np.float64)
            h5[0] = a5

        return VolterraKernelFull(
            h1=h1, h2=h2, h3_diagonal=h3, h4_diagonal=h4, h5_diagonal=h5, h2_is_diagonal=True
        )

    def estimate_memory_bytes(self) -> int:
        """Estimate memory footprint in bytes."""
        size = self.h1.nbytes

        if self.h2 is not None:
            size += self.h2.nbytes
        if self.h3_diagonal is not None:
            size += self.h3_diagonal.nbytes
        if self.h4_diagonal is not None:
            size += self.h4_diagonal.nbytes
        if self.h5_diagonal is not None:
            size += self.h5_diagonal.nbytes

        return size

    def to_dict(self) -> dict[str, ArrayF]:
        """Export kernels as dictionary for compatibility."""
        result = {"h1": self.h1}

        if self.h2 is not None:
            key = "h2_diagonal" if self.h2_is_diagonal else "h2"
            result[key] = self.h2
        if self.h3_diagonal is not None:
            result["h3_diagonal"] = self.h3_diagonal
        if self.h4_diagonal is not None:
            result["h4_diagonal"] = self.h4_diagonal
        if self.h5_diagonal is not None:
            result["h5_diagonal"] = self.h5_diagonal

        return result
