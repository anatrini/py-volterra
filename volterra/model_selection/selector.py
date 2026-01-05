"""
Automatic model selection for Volterra system identification.

Selects between Diagonal Memory Polynomial, Generalized Memory Polynomial,
and full TT-Volterra based on validation performance and information criteria.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from volterra.models.gmp import GeneralizedMemoryPolynomial, GMPConfig
from volterra.models.tt_volterra_full import TTVolterraFullConfig, TTVolterraMIMO
from volterra.utils.shapes import canonicalize_input, canonicalize_output, validate_mimo_data


@dataclass
class ModelSelectionConfig:
    """Configuration for automatic model selection.

    Attributes:
        memory_length: Number of delay taps for all models.
        order: Polynomial order for all models.
        validation_split: Fraction of data to use for validation (if not provided explicitly).
        selection_criterion: Criterion for model selection ('aic', 'bic', 'nmse').
        prefer_simpler: Prefer simpler models when criterion values are close.
        simplicity_threshold: Relative threshold for preferring simpler model.
        try_diagonal: Whether to try diagonal Memory Polynomial.
        try_gmp: Whether to try Generalized Memory Polynomial.
        try_tt_full: Whether to try full TT-Volterra.
        gmp_max_cross_lag: Max cross-lag distance for GMP.
        gmp_max_cross_order: Max cross-order for GMP.
        tt_ranks: TT ranks for full TT-Volterra (None = auto).
        verbose: Print fitting progress.
    """

    memory_length: int = 5
    order: int = 3
    validation_split: float = 0.2
    selection_criterion: str = "bic"
    prefer_simpler: bool = True
    simplicity_threshold: float = 0.02
    try_diagonal: bool = True
    try_gmp: bool = True
    try_tt_full: bool = False
    gmp_max_cross_lag: int = 2
    gmp_max_cross_order: int = 2
    tt_ranks: list | None = None
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.memory_length <= 0:
            raise ValueError("memory_length must be positive")
        if self.order < 1:
            raise ValueError("order must be at least 1")
        if not (0 < self.validation_split < 1):
            raise ValueError("validation_split must be in (0, 1)")
        if self.selection_criterion not in ["aic", "bic", "nmse"]:
            raise ValueError("selection_criterion must be 'aic', 'bic', or 'nmse'")
        if not (self.try_diagonal or self.try_gmp or self.try_tt_full):
            raise ValueError("At least one model type must be enabled")


class ModelSelector:
    """Automatic model selection for Volterra system identification.

    Fits multiple models of increasing complexity and selects the best based on
    validation performance using information criteria (AIC/BIC) or NMSE.

    Model hierarchy (increasing complexity):
        1. Diagonal Memory Polynomial (MP): O(N*M) parameters
        2. Generalized Memory Polynomial (GMP): O(N*M + cross-terms)
        3. Full TT-Volterra: O(M*r^2*(I*N)^2) with rank adaptation

    Parameters:
        config: Model selection configuration (optional).
        **kwargs: Shorthand for config parameters.

    Properties:
        is_fitted: Whether selector has been fitted.
        selected_model_type: Type of selected model.

    Methods:
        fit(x, y, x_val, y_val): Fit multiple models and select best.
        predict(x): Predict using selected model.
        get_selected_model(): Get the selected model object.
        get_all_results(): Get metrics for all tried models.
        explain(): Get explanation of model selection.

    Example:
        >>> from volterra.model_selection import ModelSelector
        >>>
        >>> # Automatic model selection
        >>> selector = ModelSelector(memory_length=10, order=3)
        >>> selector.fit(x_train, y_train)
        >>>
        >>> # Predict with best model
        >>> y_pred = selector.predict(x_test)
        >>>
        >>> # Understand selection
        >>> print(selector.explain())
        >>> print(f"Selected: {selector.selected_model_type}")
    """

    def __init__(self, config: ModelSelectionConfig | None = None, **kwargs):
        """Initialize ModelSelector.

        Args:
            config: ModelSelectionConfig instance (optional).
            **kwargs: Shorthand for config parameters.
        """
        if config is None:
            config = ModelSelectionConfig(**kwargs)
        elif kwargs:
            raise ValueError("Cannot specify both config and kwargs")

        self.config = config
        self._selected_model: Any | None = None
        self._selected_type: str | None = None
        self._all_results: dict[str, dict] | None = None
        self._n_inputs: int | None = None
        self._n_outputs: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Check if selector has been fitted."""
        return self._selected_model is not None

    @property
    def selected_model_type(self) -> str:
        """Get type of selected model."""
        if not self.is_fitted:
            raise RuntimeError("ModelSelector must be fitted before accessing selected_model_type")
        return self._selected_type

    def _split_train_val(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and validation sets.

        Args:
            x: Input data.
            y: Output data.

        Returns:
            (x_train, y_train, x_val, y_val)
        """
        n_samples = x.shape[0]
        n_val = int(n_samples * self.config.validation_split)
        n_train = n_samples - n_val

        x_train = x[:n_train]
        y_train = y[:n_train]
        x_val = x[n_train:]
        y_val = y[n_train:]

        return x_train, y_train, x_val, y_val

    def _compute_nmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Normalized Mean Squared Error.

        Args:
            y_true: Ground truth output.
            y_pred: Predicted output.

        Returns:
            NMSE value.
        """
        mse = np.mean((y_true - y_pred) ** 2)
        var = np.var(y_true)
        if var < 1e-12:
            return 0.0 if mse < 1e-12 else np.inf
        return mse / var

    def _compute_aic(self, nmse: float, n_samples: int, n_params: int) -> float:
        """Compute Akaike Information Criterion.

        Args:
            nmse: Normalized mean squared error.
            n_samples: Number of samples.
            n_params: Number of model parameters.

        Returns:
            AIC value (lower is better).
        """
        # AIC = n * log(MSE) + 2 * k
        # For NMSE, MSE = NMSE * var(y), but var(y) is constant across models
        # So we use: AIC = n * log(NMSE) + 2 * k
        if nmse < 1e-12:
            nmse = 1e-12
        return n_samples * np.log(nmse) + 2 * n_params

    def _compute_bic(self, nmse: float, n_samples: int, n_params: int) -> float:
        """Compute Bayesian Information Criterion.

        Args:
            nmse: Normalized mean squared error.
            n_samples: Number of samples.
            n_params: Number of model parameters.

        Returns:
            BIC value (lower is better).
        """
        # BIC = n * log(MSE) + k * log(n)
        if nmse < 1e-12:
            nmse = 1e-12
        return n_samples * np.log(nmse) + n_params * np.log(n_samples)

    def _count_parameters(self, model: Any, model_type: str) -> int:
        """Count number of parameters in model.

        Args:
            model: Fitted model.
            model_type: Type of model.

        Returns:
            Number of parameters.
        """
        if model_type in ["Diagonal-MP", "GMP"]:
            # GMP stores coefficients per output
            return model.total_terms * model.n_inputs * model.n_outputs
        elif model_type == "TT-Full":
            # TTVolterraMIMO has total_parameters property
            return model.total_parameters
        else:
            return 0

    def _try_diagonal_mp(
        self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray
    ) -> dict[str, Any]:
        """Try diagonal Memory Polynomial.

        Args:
            x_train, y_train: Training data.
            x_val, y_val: Validation data.

        Returns:
            Dictionary with model, predictions, and metrics.
        """
        if self.config.verbose:
            print("Trying Diagonal Memory Polynomial...")

        # Create diagonal-only GMP (equivalent to MP)
        gmp_config = GMPConfig(
            max_cross_lag_distance=0, regularization=1e-8, verbose=False  # Diagonal only
        )
        model = GeneralizedMemoryPolynomial(
            memory_length=self.config.memory_length, order=self.config.order, config=gmp_config
        )

        # Fit
        model.fit(x_train, y_train)

        # Predict on validation
        y_val_pred = model.predict(x_val)

        # Compute metrics (GMP returns full-length predictions)
        nmse = self._compute_nmse(y_val, y_val_pred)
        n_params = self._count_parameters(model, "Diagonal-MP")
        aic = self._compute_aic(nmse, len(y_val), n_params)
        bic = self._compute_bic(nmse, len(y_val), n_params)

        if self.config.verbose:
            print(f"  NMSE: {nmse:.6e}, AIC: {aic:.2f}, BIC: {bic:.2f}, Params: {n_params}")

        return {
            "model": model,
            "nmse": nmse,
            "aic": aic,
            "bic": bic,
            "n_params": n_params,
            "y_val_pred": y_val_pred,
        }

    def _try_gmp(
        self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray
    ) -> dict[str, Any]:
        """Try Generalized Memory Polynomial.

        Args:
            x_train, y_train: Training data.
            x_val, y_val: Validation data.

        Returns:
            Dictionary with model, predictions, and metrics.
        """
        if self.config.verbose:
            print("Trying Generalized Memory Polynomial...")

        gmp_config = GMPConfig(
            max_cross_lag_distance=self.config.gmp_max_cross_lag,
            max_cross_order=self.config.gmp_max_cross_order,
            regularization=1e-8,
            verbose=False,
        )
        model = GeneralizedMemoryPolynomial(
            memory_length=self.config.memory_length, order=self.config.order, config=gmp_config
        )

        # Fit
        model.fit(x_train, y_train)

        # Predict on validation
        y_val_pred = model.predict(x_val)

        # Compute metrics
        nmse = self._compute_nmse(y_val, y_val_pred)
        n_params = self._count_parameters(model, "GMP")
        aic = self._compute_aic(nmse, len(y_val), n_params)
        bic = self._compute_bic(nmse, len(y_val), n_params)

        if self.config.verbose:
            print(f"  NMSE: {nmse:.6e}, AIC: {aic:.2f}, BIC: {bic:.2f}, Params: {n_params}")

        return {
            "model": model,
            "nmse": nmse,
            "aic": aic,
            "bic": bic,
            "n_params": n_params,
            "y_val_pred": y_val_pred,
        }

    def _try_tt_full(
        self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray
    ) -> dict[str, Any]:
        """Try full TT-Volterra.

        Args:
            x_train, y_train: Training data.
            x_val, y_val: Validation data.

        Returns:
            Dictionary with model, predictions, and metrics.
        """
        if self.config.verbose:
            print("Trying full TT-Volterra...")

        # Determine ranks
        if self.config.tt_ranks is None:
            # Auto: use small ranks
            ranks = [1] + [2] * (self.config.order - 1) + [1]
        else:
            ranks = self.config.tt_ranks

        tt_config = TTVolterraFullConfig(max_iter=50, tol=1e-6, regularization=1e-6, verbose=False)
        model = TTVolterraMIMO(
            memory_length=self.config.memory_length,
            order=self.config.order,
            ranks=ranks,
            config=tt_config,
        )

        # Fit
        model.fit(x_train, y_train)

        # Predict on validation
        y_val_pred = model.predict(x_val)

        # Trim y_val to match prediction length (TT models produce T - memory_length + 1 outputs)
        y_val_trimmed = y_val[self.config.memory_length - 1 :]

        # Compute metrics
        nmse = self._compute_nmse(y_val_trimmed, y_val_pred)
        n_params = self._count_parameters(model, "TT-Full")
        aic = self._compute_aic(nmse, len(y_val_trimmed), n_params)
        bic = self._compute_bic(nmse, len(y_val_trimmed), n_params)

        if self.config.verbose:
            print(f"  NMSE: {nmse:.6e}, AIC: {aic:.2f}, BIC: {bic:.2f}, Params: {n_params}")

        return {
            "model": model,
            "nmse": nmse,
            "aic": aic,
            "bic": bic,
            "n_params": n_params,
            "y_val_pred": y_val_pred,
        }

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "ModelSelector":
        """Fit multiple models and select the best.

        Args:
            x: Training input data, shape (T, I) or (T,).
            y: Training output data, shape (T, O) or (T,).
            x_val: Validation input (optional, will split if not provided).
            y_val: Validation output (optional, will split if not provided).

        Returns:
            self (for method chaining).
        """
        # Validate data
        validate_mimo_data(x, y)
        x_canon = canonicalize_input(x)
        y_canon = canonicalize_output(y)

        self._n_inputs = x_canon.shape[1]
        self._n_outputs = y_canon.shape[1]

        # Split train/val if not provided
        if x_val is None or y_val is None:
            x_train, y_train, x_val, y_val = self._split_train_val(
                x_canon if self._n_inputs > 1 else x_canon[:, 0],
                y_canon if self._n_outputs > 1 else y_canon[:, 0],
            )
        else:
            x_train = x_canon if self._n_inputs > 1 else x_canon[:, 0]
            y_train = y_canon if self._n_outputs > 1 else y_canon[:, 0]
            validate_mimo_data(x_val, y_val)
            x_val_canon = canonicalize_input(x_val)
            y_val_canon = canonicalize_output(y_val)
            x_val = x_val_canon if self._n_inputs > 1 else x_val_canon[:, 0]
            y_val = y_val_canon if self._n_outputs > 1 else y_val_canon[:, 0]

        # Try each model type
        results = {}

        if self.config.try_diagonal:
            results["Diagonal-MP"] = self._try_diagonal_mp(x_train, y_train, x_val, y_val)

        if self.config.try_gmp:
            results["GMP"] = self._try_gmp(x_train, y_train, x_val, y_val)

        if self.config.try_tt_full:
            results["TT-Full"] = self._try_tt_full(x_train, y_train, x_val, y_val)

        # Select best model
        criterion_key = self.config.selection_criterion
        if criterion_key == "nmse":
            criterion_key = "nmse"
        else:
            criterion_key = self.config.selection_criterion  # 'aic' or 'bic'

        # Find model with lowest criterion
        best_model_type = None
        best_criterion = np.inf

        # Model complexity order (for tie-breaking with prefer_simpler)
        complexity_order = ["Diagonal-MP", "GMP", "TT-Full"]

        for model_type in complexity_order:
            if model_type not in results:
                continue

            criterion_value = results[model_type][criterion_key]

            if self.config.prefer_simpler and best_model_type is not None:
                # If new model is more complex but criterion is comparable, keep simpler
                relative_improvement = (best_criterion - criterion_value) / abs(
                    best_criterion + 1e-12
                )
                if relative_improvement < self.config.simplicity_threshold:
                    # Not enough improvement to justify complexity
                    continue

            if criterion_value < best_criterion:
                best_criterion = criterion_value
                best_model_type = model_type

        # Store results
        self._selected_type = best_model_type
        self._selected_model = results[best_model_type]["model"]
        self._all_results = results

        if self.config.verbose:
            print(f"\nSelected: {best_model_type}")

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict output using selected model.

        Args:
            x: Input data, shape (T, I) or (T,).

        Returns:
            Predicted output, shape (T, O) or (T,).

        Raises:
            RuntimeError: If selector not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("ModelSelector must be fitted before prediction")

        return self._selected_model.predict(x)

    def get_selected_model(self) -> Any:
        """Get the selected model object.

        Returns:
            Fitted model (GeneralizedMemoryPolynomial or TTVolterraMIMO).

        Raises:
            RuntimeError: If selector not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("ModelSelector must be fitted before accessing model")

        return self._selected_model

    def get_all_results(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all tried models.

        Returns:
            Dictionary mapping model type to metrics dict.

        Raises:
            RuntimeError: If selector not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("ModelSelector must be fitted before accessing results")

        # Return copy without y_val_pred (too large)
        results_clean = {}
        for model_type, result in self._all_results.items():
            results_clean[model_type] = {
                "nmse": result["nmse"],
                "aic": result["aic"],
                "bic": result["bic"],
                "n_params": result["n_params"],
            }

        return results_clean

    def explain(self) -> str:
        """Get explanation of model selection process.

        Returns:
            Multi-line string explaining selection.

        Raises:
            RuntimeError: If selector not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("ModelSelector must be fitted before explain")

        lines = []
        lines.append("=" * 70)
        lines.append("MODEL SELECTION REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Selection criterion: {self.config.selection_criterion.upper()}")
        lines.append(f"Prefer simpler models: {self.config.prefer_simpler}")
        lines.append("")
        lines.append("-" * 70)
        lines.append("MODEL COMPARISON")
        lines.append("-" * 70)
        lines.append(f"{'Model':<20} {'NMSE':>12} {'AIC':>12} {'BIC':>12} {'Params':>10}")
        lines.append("-" * 70)

        for model_type in ["Diagonal-MP", "GMP", "TT-Full"]:
            if model_type not in self._all_results:
                continue

            result = self._all_results[model_type]
            selected_marker = " (*)" if model_type == self._selected_type else ""

            lines.append(
                f"{model_type + selected_marker:<20} "
                f"{result['nmse']:>12.6e} "
                f"{result['aic']:>12.2f} "
                f"{result['bic']:>12.2f} "
                f"{result['n_params']:>10d}"
            )

        lines.append("-" * 70)
        lines.append("")
        lines.append(f"Selected model: {self._selected_type}")
        lines.append("")

        # Add interpretation
        selected_result = self._all_results[self._selected_type]
        lines.append("INTERPRETATION:")
        lines.append(f"  - Validation NMSE: {selected_result['nmse']:.6e}")
        lines.append(f"  - Model parameters: {selected_result['n_params']}")

        if self.config.prefer_simpler and len(self._all_results) > 1:
            lines.append("  - Simpler models preferred when performance is comparable")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted=True" if self.is_fitted else "fitted=False"
        if self.is_fitted:
            return f"ModelSelector({fitted_str}, selected={self._selected_type})"
        else:
            return f"ModelSelector({fitted_str})"
