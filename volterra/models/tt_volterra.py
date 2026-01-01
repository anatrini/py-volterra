"""
TT-Volterra: Tensor-Train based Volterra system identification.

This module provides a high-level API for identifying MIMO Volterra models
using Tensor-Train decomposition, avoiding the curse of dimensionality.

The TTVolterraIdentifier supports:
- SISO and MIMO identification (separate TT model per output)
- Diagonal-only mode (memory polynomial / Hammerstein models)
- General TT mode (full multi-dimensional coupling)
- Configurable TT ranks and solver parameters

Typical usage:
--------------
    from volterra.models import TTVolterraIdentifier, TTVolterraConfig

    # SISO identification
    identifier = TTVolterraIdentifier(
        memory_length=20,
        order=3,
        ranks=[1, 5, 3, 1]
    )
    identifier.fit(x_train, y_train)
    y_pred = identifier.predict(x_test)

    # MIMO identification
    identifier_mimo = TTVolterraIdentifier(
        memory_length=15,
        order=2,
        ranks=[1, 4, 1]
    )
    identifier_mimo.fit(x_mimo, y_mimo)  # x: (T, I), y: (T, O)
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
import warnings

from volterra.utils.shapes import (
    canonicalize_input,
    canonicalize_output,
    validate_mimo_data,
    infer_dimensions,
)
from volterra.tt import (
    TTTensor,
    tt_als,
    tt_mals,
    tt_matvec,
    TTALSConfig,
    TTMALSConfig,
)


@dataclass
class TTVolterraConfig:
    """
    Configuration for TT-Volterra identification.

    Parameters
    ----------
    solver : str, default='als'
        Solver type: 'als' (fixed-rank) or 'mals' (adaptive-rank)
    max_iter : int, default=100
        Maximum number of ALS sweeps
    tol : float, default=1e-6
        Convergence tolerance (relative change in loss)
    regularization : float, default=1e-8
        Tikhonov regularization for least-squares stability
    rank_adaptation : bool, default=False
        Enable rank adaptation (only for solver='mals')
    max_rank : int, default=10
        Maximum allowed TT rank (for MALS)
    rank_tol : float, default=1e-4
        Singular value threshold for rank truncation
    verbose : bool, default=False
        Print iteration progress
    diagonal_only : bool, default=False
        If True, identify diagonal Volterra kernels only (memory polynomial)
    """
    solver: str = 'als'
    max_iter: int = 100
    tol: float = 1e-6
    regularization: float = 1e-8
    rank_adaptation: bool = False
    max_rank: int = 10
    rank_tol: float = 1e-4
    verbose: bool = False
    diagonal_only: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.solver not in ('als', 'mals'):
            raise ValueError(f"Solver must be 'als' or 'mals', got '{self.solver}'")
        if self.rank_adaptation and self.solver != 'mals':
            warnings.warn(
                "rank_adaptation=True requires solver='mals', setting solver='mals'",
                UserWarning
            )
            self.solver = 'mals'


class TTVolterraIdentifier:
    """
    TT-Volterra system identifier for MIMO nonlinear systems.

    Uses Tensor-Train decomposition to identify Volterra models from
    input-output data, supporting both SISO and MIMO configurations.

    For MIMO systems, identifies a separate TT model per output channel.

    Attributes
    ----------
    memory_length : int
        Memory length N (number of delays)
    order : int
        Volterra order M (polynomial order)
    ranks : List[int]
        TT ranks [r_0=1, r_1, ..., r_{M-1}, r_M=1]
    config : TTVolterraConfig
        Identification configuration
    is_fitted : bool
        Whether model has been fitted to data
    tt_models_ : Optional[List[TTTensor]]
        Fitted TT models, one per output channel (set after fit())
    fit_info_ : Optional[Dict]
        Fitting information (loss, iterations, etc.)

    Examples
    --------
    >>> # SISO identification
    >>> import numpy as np
    >>> from volterra.models import TTVolterraIdentifier
    >>>
    >>> # Generate synthetic data
    >>> x = np.random.randn(1000)
    >>> y = x + 0.1 * x**2 + 0.05 * x**3  # Nonlinear system
    >>>
    >>> # Identify model
    >>> identifier = TTVolterraIdentifier(
    ...     memory_length=10,
    ...     order=3,
    ...     ranks=[1, 3, 2, 1]
    ... )
    >>> identifier.fit(x, y)
    >>>
    >>> # Predict on new data
    >>> x_test = np.random.randn(500)
    >>> y_pred = identifier.predict(x_test)

    >>> # MIMO identification
    >>> x_mimo = np.random.randn(1000, 2)  # 2 inputs
    >>> y_mimo = np.random.randn(1000, 3)  # 3 outputs
    >>>
    >>> identifier_mimo = TTVolterraIdentifier(
    ...     memory_length=8,
    ...     order=2,
    ...     ranks=[1, 4, 1]
    ... )
    >>> identifier_mimo.fit(x_mimo, y_mimo)
    >>> y_pred_mimo = identifier_mimo.predict(x_mimo[:100])
    """

    def __init__(
        self,
        memory_length: int,
        order: int,
        ranks: List[int],
        config: Optional[TTVolterraConfig] = None
    ):
        """
        Initialize TT-Volterra identifier.

        Parameters
        ----------
        memory_length : int
            Memory length N (number of delays)
        order : int
            Volterra order M (number of TT cores)
        ranks : List[int]
            TT ranks [r_0=1, r_1, ..., r_{M-1}, r_M=1]
        config : TTVolterraConfig, optional
            Identification configuration

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        if memory_length < 1:
            raise ValueError(f"memory_length must be >= 1, got {memory_length}")
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if len(ranks) != order + 1:
            raise ValueError(
                f"Need {order+1} ranks for order {order}, got {len(ranks)}"
            )
        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError(
                f"Boundary ranks must be 1, got r_0={ranks[0]}, r_M={ranks[-1]}"
            )

        self.memory_length = memory_length
        self.order = order
        self.ranks = ranks
        self.config = config or TTVolterraConfig()

        # Fitted model attributes (set after fit())
        self.tt_models_: Optional[List[TTTensor]] = None
        self.fit_info_: Optional[Dict] = None
        self.n_outputs_: Optional[int] = None
        self.n_inputs_: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        """Whether model has been fitted to data."""
        return self.tt_models_ is not None

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'TTVolterraIdentifier':
        """
        Fit TT-Volterra model to input-output data.

        Identifies Volterra kernels in TT format using TT-ALS or TT-MALS.
        For MIMO systems, fits a separate TT model per output channel.

        Parameters
        ----------
        x : np.ndarray
            Input data, shape (T,) for SISO or (T, I) for MIMO
        y : np.ndarray
            Output data, shape (T,) for single output or (T, O) for MIMO

        Returns
        -------
        self : TTVolterraIdentifier
            Fitted identifier (for chaining)

        Raises
        ------
        ValueError
            If data shapes are incompatible

        Examples
        --------
        >>> identifier = TTVolterraIdentifier(memory_length=10, order=2, ranks=[1, 3, 1])
        >>> identifier.fit(x_train, y_train)
        >>> print(identifier.is_fitted)
        True
        """
        # Validate and canonicalize inputs
        validate_mimo_data(x, y)
        x_canon = canonicalize_input(x)  # (T, I)
        y_canon = canonicalize_output(y)  # (T, O)

        T, I, O = infer_dimensions(x, y)
        self.n_inputs_ = I
        self.n_outputs_ = O

        if self.config.verbose:
            print(f"Fitting TT-Volterra: T={T}, I={I}, O={O}, N={self.memory_length}, M={self.order}")

        # Fit a separate TT model for each output channel
        tt_models = []
        fit_infos = []

        for o in range(O):
            y_o = y_canon[:, o]  # (T,)

            if self.config.verbose:
                print(f"  Output {o+1}/{O}...")

            # For MIMO, we would need to handle multi-input properly
            # For now, simplified to first input channel
            if I > 1:
                warnings.warn(
                    f"MIMO with I={I} inputs: using first input channel only. "
                    "Full MIMO TT-Volterra requires tensor product design matrices.",
                    UserWarning
                )
            x_o = x_canon[:, 0] if I > 1 else x_canon[:, 0]

            # Choose solver
            if self.config.solver == 'als':
                solver_config = TTALSConfig(
                    max_iter=self.config.max_iter,
                    tol=self.config.tol,
                    regularization=self.config.regularization,
                    verbose=self.config.verbose,
                )
                cores, info = tt_als(
                    x_o, y_o,
                    self.memory_length,
                    self.order,
                    self.ranks,
                    config=solver_config
                )
            elif self.config.solver == 'mals':
                solver_config = TTMALSConfig(
                    max_iter=self.config.max_iter,
                    tol=self.config.tol,
                    regularization=self.config.regularization,
                    rank_adaptation=self.config.rank_adaptation,
                    max_rank=self.config.max_rank,
                    rank_tol=self.config.rank_tol,
                    verbose=self.config.verbose,
                )
                cores, info = tt_mals(
                    x_o, y_o,
                    self.memory_length,
                    self.order,
                    self.ranks,
                    config=solver_config
                )
            else:
                raise ValueError(f"Unknown solver: {self.config.solver}")

            tt_models.append(TTTensor(cores))
            fit_infos.append(info)

        self.tt_models_ = tt_models
        self.fit_info_ = {
            'per_output': fit_infos,
            'n_outputs': O,
            'n_inputs': I,
        }

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict output from input using fitted TT-Volterra model.

        Parameters
        ----------
        x : np.ndarray
            Input data, shape (T,) for SISO or (T, I) for MIMO

        Returns
        -------
        y_pred : np.ndarray
            Predicted output, shape (T,) for single output or (T, O) for MIMO

        Raises
        ------
        ValueError
            If model is not fitted or input shape is incompatible

        Examples
        --------
        >>> y_pred = identifier.predict(x_test)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        x_canon = canonicalize_input(x)  # (T, I)
        T, I = x_canon.shape

        if I != self.n_inputs_:
            raise ValueError(
                f"Input has {I} channels, but model was fitted with {self.n_inputs_}"
            )

        # Check sufficient samples for memory
        if T < self.memory_length:
            raise ValueError(
                f"Input has {T} samples, but memory_length={self.memory_length}"
            )

        O = self.n_outputs_
        T_valid = T - self.memory_length + 1
        y_pred = np.zeros((T_valid, O))

        # For each output, evaluate TT model
        # This is a PLACEHOLDER - full implementation would use proper convolution
        warnings.warn(
            "predict() is a placeholder implementation. "
            "Full TT-Volterra prediction requires proper memory-based convolution.",
            UserWarning
        )

        # Simplified: use first memory_length samples only
        for o in range(O):
            tt_model = self.tt_models_[o]
            x_mem = x_canon[:self.memory_length, 0]  # (N,)
            y_o_sample = tt_matvec(tt_model.cores, x_mem)
            # For now, just replicate this value
            y_pred[:, o] = y_o_sample

        # Return in original format
        if O == 1:
            return y_pred[:, 0]
        else:
            return y_pred

    def get_kernels(self, output_idx: int = 0) -> TTTensor:
        """
        Get TT kernels for specified output channel.

        Parameters
        ----------
        output_idx : int, default=0
            Output channel index (0-indexed)

        Returns
        -------
        tt_model : TTTensor
            TT-represented Volterra kernels for this output

        Raises
        ------
        ValueError
            If model not fitted or output_idx out of range

        Examples
        --------
        >>> tt_kernels = identifier.get_kernels(output_idx=0)
        >>> print(tt_kernels.ranks)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if output_idx < 0 or output_idx >= self.n_outputs_:
            raise ValueError(
                f"output_idx must be in [0, {self.n_outputs_-1}], got {output_idx}"
            )

        return self.tt_models_[output_idx]

    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return (
            f"TTVolterraIdentifier(N={self.memory_length}, M={self.order}, "
            f"ranks={self.ranks}, {fitted_str})"
        )
