"""
Full TT-Volterra MIMO system identification with arbitrary ranks.

This module provides a high-level API for general (non-diagonal) MIMO Volterra
identification using Tensor-Train decomposition with arbitrary ranks.

Compared to diagonal Volterra (memory polynomial), this enables modeling of
cross-memory interactions at the cost of higher computational complexity.

Typical usage:
--------------
    from volterra.models import TTVolterraMIMO, TTVolterraFullConfig

    # Configure general TT-Volterra
    identifier = TTVolterraMIMO(
        memory_length=10,
        order=3,
        ranks=[1, 3, 2, 1],  # Non-diagonal ranks
        config=TTVolterraFullConfig(
            max_iter=100,
            tol=1e-6,
            regularization=1e-8
        )
    )

    # Fit to data
    identifier.fit(x_train, y_train)

    # Predict
    y_pred = identifier.predict(x_test)

    # Diagnostics
    print(identifier.diagnostics())
"""

from dataclasses import dataclass

import numpy as np

from volterra.tt.tt_als_mimo import (
    build_mimo_delay_matrix,
    evaluate_tt_volterra_mimo,
    tt_als_full_mimo,
)
from volterra.utils.shapes import (
    canonicalize_input,
    canonicalize_output,
    infer_dimensions,
    validate_mimo_data,
)


@dataclass
class TTVolterraFullConfig:
    """
    Configuration for full TT-Volterra identification.

    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of ALS sweeps
    tol : float, default=1e-6
        Convergence tolerance (relative change in loss)
    regularization : float, default=1e-8
        Tikhonov regularization for numerical stability
    verbose : bool, default=False
        Print iteration progress
    check_monotonic : bool, default=True
        Enforce monotonic loss decrease (reject bad updates)
    """

    max_iter: int = 100
    tol: float = 1e-6
    regularization: float = 1e-8
    verbose: bool = False
    check_monotonic: bool = True


class TTVolterraMIMO:
    """
    Full TT-Volterra MIMO system identifier with arbitrary ranks.

    Identifies general (non-diagonal) Volterra models using Tensor-Train
    decomposition, supporting arbitrary TT ranks for modeling cross-memory
    interactions.

    This is the most general Volterra identifier, suitable for:
    - Complex nonlinear systems with memory coupling
    - Offline/analysis-grade identification (not real-time)
    - Systems where diagonal Volterra is insufficient

    For diagonal-only (memory polynomial) systems, use TTVolterraIdentifier
    instead for better performance.

    Attributes
    ----------
    memory_length : int
        Memory length N (number of delays per input channel)
    order : int
        Volterra order M (polynomial degree)
    ranks : List[int]
        TT ranks [r_0=1, r_1, ..., r_{M-1}, r_M=1]
    config : TTVolterraFullConfig
        Identification configuration
    is_fitted : bool
        Whether model has been fitted to data
    cores_ : Optional[List[List[np.ndarray]]]
        Fitted TT cores, one list per output channel (set after fit())
    fit_info_ : Optional[Dict]
        Fitting information (loss, iterations, etc.)

    Examples
    --------
    >>> # SISO with rank-2 TT
    >>> import numpy as np
    >>> from volterra.models import TTVolterraMIMO
    >>>
    >>> x = np.random.randn(1000)
    >>> y = x + 0.1 * x**2 + 0.05 * x**3  # Nonlinear system
    >>>
    >>> identifier = TTVolterraMIMO(
    ...     memory_length=10,
    ...     order=3,
    ...     ranks=[1, 2, 2, 1]
    ... )
    >>> identifier.fit(x, y)
    >>> y_pred = identifier.predict(x[:500])

    >>> # MIMO with 2 inputs
    >>> x_mimo = np.random.randn(1000, 2)
    >>> y_mimo = nonlinear_mimo_system(x_mimo)
    >>>
    >>> identifier_mimo = TTVolterraMIMO(
    ...     memory_length=8,
    ...     order=2,
    ...     ranks=[1, 3, 1]
    ... )
    >>> identifier_mimo.fit(x_mimo, y_mimo)
    >>> y_pred = identifier_mimo.predict(x_mimo[:500])

    References
    ----------
    - Batselier, Chen, Wong (2017), "Tensor Network alternating linear scheme
      for MIMO Volterra system identification", Automatica, Vol. 84, pp. 26-35
    - Oseledets (2011), "Tensor-Train Decomposition", SIAM J. Sci. Comput.
    """

    def __init__(
        self,
        memory_length: int,
        order: int,
        ranks: list[int],
        config: TTVolterraFullConfig | None = None,
    ):
        """
        Initialize full TT-Volterra identifier.

        Parameters
        ----------
        memory_length : int
            Memory length N (number of delays per input channel)
        order : int
            Volterra order M (number of TT cores)
        ranks : List[int]
            TT ranks [r_0=1, r_1, ..., r_{M-1}, r_M=1]
            For diagonal Volterra, use all ranks = 1
        config : TTVolterraFullConfig, optional
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
            raise ValueError(f"Need {order+1} ranks for order {order}, got {len(ranks)}")
        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError(f"Boundary ranks must be 1, got r_0={ranks[0]}, r_M={ranks[-1]}")

        self.memory_length = memory_length
        self.order = order
        self.ranks = ranks
        self.config = config or TTVolterraFullConfig()

        # Fitted model attributes (set after fit())
        self.cores_: list[list[np.ndarray]] | None = None
        self.fit_info_: dict | None = None
        self.n_outputs_: int | None = None
        self.n_inputs_: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether model has been fitted to data."""
        return self.cores_ is not None

    @property
    def is_diagonal(self) -> bool:
        """Whether this is a diagonal TT (all ranks = 1)."""
        return all(r == 1 for r in self.ranks)

    @property
    def total_parameters(self) -> int:
        """Total number of parameters in the TT model."""
        if not self.is_fitted:
            # Estimate from ranks
            n_inputs = self.n_inputs_ or 1
            mode_size = n_inputs * self.memory_length
            n_params = 0
            for k in range(self.order):
                r_left = self.ranks[k]
                r_right = self.ranks[k + 1]
                n_params += r_left * mode_size * r_right
            return n_params
        else:
            # Count from actual cores
            n_params = 0
            for cores_o in self.cores_:
                for core in cores_o:
                    n_params += core.size
            return n_params

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TTVolterraMIMO":
        """
        Fit full TT-Volterra model to input-output data.

        Identifies general Volterra kernels in TT format using TT-ALS.
        For MIMO systems with multiple outputs, fits a separate TT model
        per output channel.

        Parameters
        ----------
        x : np.ndarray
            Input data, shape (T,) for SISO or (T, I) for MIMO
        y : np.ndarray
            Output data, shape (T,) for single output or (T, O) for MIMO

        Returns
        -------
        self : TTVolterraMIMO
            Fitted identifier (for chaining)

        Raises
        ------
        ValueError
            If data shapes are incompatible

        Examples
        --------
        >>> identifier = TTVolterraMIMO(memory_length=10, order=2, ranks=[1, 3, 1])
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
            print(
                f"Fitting full TT-Volterra: T={T}, I={I}, O={O}, N={self.memory_length}, M={self.order}"
            )
            print(f"Ranks: {self.ranks} (diagonal: {self.is_diagonal})")
            print(f"Total parameters (estimated): {self.total_parameters}")

        # Fit a separate TT model for each output channel
        cores_all = []
        fit_infos = []

        for o in range(O):
            y_o = y_canon[:, o]  # (T,)

            if self.config.verbose:
                print(f"  Output {o+1}/{O}...")

            # Prepare input (SISO or MIMO)
            x_o = x_canon if I > 1 else x_canon[:, 0]

            # Fit using full TT-ALS
            cores, info = tt_als_full_mimo(
                x_o,
                y_o,
                memory_length=self.memory_length,
                order=self.order,
                ranks=self.ranks,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                regularization=self.config.regularization,
                verbose=self.config.verbose,
                check_monotonic=self.config.check_monotonic,
            )

            cores_all.append(cores)
            fit_infos.append(info)

        self.cores_ = cores_all
        self.fit_info_ = {
            "per_output": fit_infos,
            "n_outputs": O,
            "n_inputs": I,
        }

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict output from input using fitted full TT-Volterra model.

        Parameters
        ----------
        x : np.ndarray
            Input data, shape (T,) for SISO or (T, I) for MIMO

        Returns
        -------
        y_pred : np.ndarray
            Predicted output, shape (T_valid,) for single output or
            (T_valid, O) for MIMO, where T_valid = T - N + 1

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

        if self.n_inputs_ != I:
            raise ValueError(f"Input has {I} channels, but model was fitted with {self.n_inputs_}")

        # Check sufficient samples for memory
        if self.memory_length > T:
            raise ValueError(f"Input has {T} samples, but memory_length={self.memory_length}")

        O = self.n_outputs_
        T_valid = T - self.memory_length + 1
        y_pred = np.zeros((T_valid, O))

        # Build delay matrix once for all outputs
        X_delay = build_mimo_delay_matrix(x_canon, self.memory_length)

        # For each output, evaluate TT model
        for o in range(O):
            cores = self.cores_[o]
            y_pred[:, o] = evaluate_tt_volterra_mimo(cores, X_delay)

        # Return in original format
        if O == 1:
            return y_pred[:, 0]
        else:
            return y_pred

    def get_cores(self, output_idx: int = 0) -> list[np.ndarray]:
        """
        Get TT cores for specified output channel.

        Parameters
        ----------
        output_idx : int, default=0
            Output channel index (0-indexed)

        Returns
        -------
        cores : List[np.ndarray]
            TT cores for this output

        Raises
        ------
        ValueError
            If model not fitted or output_idx out of range

        Examples
        --------
        >>> cores = identifier.get_cores(output_idx=0)
        >>> print([core.shape for core in cores])
        [(1, 10, 3), (3, 10, 2), (2, 10, 1)]
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if output_idx < 0 or output_idx >= self.n_outputs_:
            raise ValueError(f"output_idx must be in [0, {self.n_outputs_-1}], got {output_idx}")

        return self.cores_[output_idx]

    def diagnostics(self, output_idx: int = 0) -> dict:
        """
        Get diagnostic information for specified output.

        Returns training curves, convergence info, and model statistics.

        Parameters
        ----------
        output_idx : int, default=0
            Output channel index

        Returns
        -------
        diagnostics : Dict
            Dictionary with diagnostic information:
            - 'loss_history': list of losses per iteration
            - 'converged': whether convergence criterion met
            - 'final_loss': final MSE loss
            - 'iterations': number of iterations performed
            - 'ranks': TT ranks
            - 'total_parameters': total number of parameters
            - 'condition_history': condition numbers (if available)

        Examples
        --------
        >>> diag = identifier.diagnostics()
        >>> print(f"Converged: {diag['converged']}")
        >>> print(f"Final loss: {diag['final_loss']:.6e}")
        >>> plt.semilogy(diag['loss_history'])
        >>> plt.show()
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if output_idx < 0 or output_idx >= self.n_outputs_:
            raise ValueError(f"output_idx must be in [0, {self.n_outputs_-1}], got {output_idx}")

        fit_info = self.fit_info_["per_output"][output_idx]
        cores = self.cores_[output_idx]

        # Extract ranks from cores
        actual_ranks = [cores[0].shape[0]]
        for core in cores:
            actual_ranks.append(core.shape[2])

        # Count parameters
        n_params = sum(core.size for core in cores)

        diagnostics = {
            "loss_history": fit_info.get("loss_history", []),
            "converged": fit_info.get("converged", False),
            "final_loss": fit_info.get("final_loss", np.inf),
            "iterations": fit_info.get("iterations", 0),
            "ranks": actual_ranks,
            "total_parameters": n_params,
            "condition_history": fit_info.get("condition_history", []),
            "mimo": fit_info.get("mimo", False),
            "mode_size": fit_info.get("mode_size", self.memory_length),
        }

        return diagnostics

    def export_model(self, output_idx: int = 0) -> dict:
        """
        Export model parameters for C++/Rust/external implementation.

        Returns a dictionary with all necessary information to reconstruct
        and evaluate the TT-Volterra model externally.

        Parameters
        ----------
        output_idx : int, default=0
            Output channel index

        Returns
        -------
        model_export : Dict
            Dictionary with model parameters:
            - 'cores': list of numpy arrays (TT cores)
            - 'ranks': list of TT ranks
            - 'memory_length': int
            - 'order': int
            - 'n_inputs': int
            - 'metadata': dict with additional info

        Examples
        --------
        >>> model_data = identifier.export_model()
        >>> np.savez('model.npz', **{f'core_{i}': c for i, c in enumerate(model_data['cores'])})
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        cores = self.get_cores(output_idx)
        ranks = [cores[0].shape[0]] + [c.shape[2] for c in cores]

        export = {
            "cores": cores,
            "ranks": ranks,
            "memory_length": self.memory_length,
            "order": self.order,
            "n_inputs": self.n_inputs_,
            "metadata": {
                "model_type": "full_tt_volterra",
                "diagonal": self.is_diagonal,
                "total_parameters": sum(c.size for c in cores),
                "fit_info": self.fit_info_["per_output"][output_idx],
            },
        }

        return export

    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        diagonal_str = " (diagonal)" if self.is_diagonal else ""
        return (
            f"TTVolterraMIMO(N={self.memory_length}, M={self.order}, "
            f"ranks={self.ranks}{diagonal_str}, {fitted_str})"
        )
