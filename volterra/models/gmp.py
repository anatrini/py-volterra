"""
Generalized Memory Polynomial (GMP) for nonlinear system identification.

The GMP model is intermediate between diagonal Memory Polynomial (MP) and
full TT-Volterra. It includes diagonal memory polynomial terms plus selective
cross-memory interaction terms.

Reference:
    Morgan, D. R., Ma, Z., Kim, J., Zierdt, M. G., & Pastalan, J. (2006).
    "A generalized memory polynomial model for digital predistortion of RF
    power amplifiers". IEEE Transactions on Signal Processing, 54(10), 3852-3860.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np

from volterra.utils.shapes import (
    validate_mimo_data,
    canonicalize_input,
    canonicalize_output,
)
from volterra.tt.tt_als_mimo import build_mimo_delay_matrix


@dataclass
class GMPConfig:
    """Configuration for Generalized Memory Polynomial model.

    Attributes:
        max_cross_lag_distance: Maximum lag distance for cross-memory terms.
            If 0, only diagonal terms are included (equivalent to MP).
            If k > 0, includes terms x(t-k1)^p * x(t-k2)^q where |k1-k2| <= k.
        max_cross_order: Maximum total order for cross-terms.
            E.g., max_cross_order=2 allows x(t-k1) * x(t-k2) but not higher products.
        include_lead_terms: Include leading cross-memory terms (k1 < k2).
        include_lag_terms: Include lagging cross-memory terms (k1 > k2).
        regularization: Tikhonov regularization parameter (ridge regression).
        verbose: Print fitting progress.
    """
    max_cross_lag_distance: int = 0
    max_cross_order: int = 2
    include_lead_terms: bool = True
    include_lag_terms: bool = True
    regularization: float = 1e-8
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_cross_lag_distance < 0:
            raise ValueError("max_cross_lag_distance must be non-negative")
        if self.max_cross_order < 1:
            raise ValueError("max_cross_order must be at least 1")
        if self.regularization < 0:
            raise ValueError("regularization must be non-negative")


class GeneralizedMemoryPolynomial:
    """Generalized Memory Polynomial (GMP) MIMO system identifier.

    The GMP model extends the diagonal Memory Polynomial by including selective
    cross-memory interaction terms. This provides intermediate complexity between
    diagonal MP (fast, limited expressiveness) and full Volterra (expressive, slow).

    Model structure:
        Diagonal terms: sum_m sum_k h_m[k] * x(t-k)^m
        Cross-terms: sum_{k1,k2} c_{k1,k2} * x(t-k1)^p * x(t-k2)^q
            where |k1-k2| <= max_cross_lag_distance and p+q <= max_cross_order

    Parameters:
        memory_length: Number of delay taps (N).
        order: Polynomial order (M).
        config: GMP configuration (optional).

    Properties:
        is_fitted: Whether model has been fitted to data.
        is_diagonal: Whether model only has diagonal terms (no cross-terms).
        total_terms: Total number of terms in the model.
        n_inputs: Number of input channels (after fitting).
        n_outputs: Number of output channels (after fitting).

    Methods:
        fit(x, y): Fit model using regularized least squares.
        predict(x): Predict output for new input.
        get_coefficients(output_idx): Get coefficients organized by term type.
        export_model(output_idx): Export model for C++/Rust porting.

    Example:
        >>> from volterra.models import GeneralizedMemoryPolynomial, GMPConfig
        >>>
        >>> # Create GMP with cross-memory terms
        >>> config = GMPConfig(max_cross_lag_distance=2, max_cross_order=2)
        >>> model = GeneralizedMemoryPolynomial(memory_length=10, order=3, config=config)
        >>>
        >>> # Fit to data
        >>> model.fit(x_train, y_train)
        >>>
        >>> # Predict
        >>> y_pred = model.predict(x_test)
        >>>
        >>> # Export for deployment
        >>> export = model.export_model()
    """

    def __init__(
        self,
        memory_length: int,
        order: int,
        config: Optional[GMPConfig] = None
    ):
        """Initialize GMP model.

        Args:
            memory_length: Number of delay taps (N >= 1).
            order: Polynomial order (M >= 1).
            config: GMP configuration (uses defaults if None).

        Raises:
            ValueError: If parameters are invalid.
        """
        if memory_length <= 0:
            raise ValueError("memory_length must be positive")
        if order < 1:
            raise ValueError("order must be at least 1")

        self.memory_length = memory_length
        self.order = order
        self.config = config if config is not None else GMPConfig()

        # Will be set during fit()
        self._coefficients: Optional[List[np.ndarray]] = None
        self._n_inputs: Optional[int] = None
        self._n_outputs: Optional[int] = None
        self._term_indices: Optional[Dict[str, np.ndarray]] = None

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._coefficients is not None

    @property
    def is_diagonal(self) -> bool:
        """Check if model only has diagonal terms (no cross-terms)."""
        return self.config.max_cross_lag_distance == 0

    @property
    def n_inputs(self) -> int:
        """Number of input channels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before accessing n_inputs")
        return self._n_inputs

    @property
    def n_outputs(self) -> int:
        """Number of output channels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before accessing n_outputs")
        return self._n_outputs

    @property
    def total_terms(self) -> int:
        """Total number of terms in the model."""
        n_diagonal = self.memory_length * self.order

        if self.is_diagonal:
            return n_diagonal

        # Count cross-memory terms
        n_cross = 0
        for k1 in range(self.memory_length):
            for k2 in range(self.memory_length):
                lag_dist = abs(k1 - k2)
                if lag_dist == 0 or lag_dist > self.config.max_cross_lag_distance:
                    continue

                # Check lead/lag inclusion
                if k1 < k2 and not self.config.include_lead_terms:
                    continue
                if k1 > k2 and not self.config.include_lag_terms:
                    continue

                # Count valid order combinations
                for p in range(1, self.order + 1):
                    for q in range(1, self.order + 1):
                        if p + q <= self.config.max_cross_order:
                            n_cross += 1

        return n_diagonal + n_cross

    def _generate_term_indices(self, I: int) -> Dict[str, List[Tuple]]:
        """Generate indices for all terms in the model.

        Args:
            I: Number of input channels.

        Returns:
            Dictionary with 'diagonal' and 'cross_terms' keys, each containing
            list of tuples describing the terms.
        """
        indices = {'diagonal': [], 'cross_terms': []}

        # Diagonal terms: (input_idx, lag_idx, order)
        for i in range(I):
            for k in range(self.memory_length):
                for m in range(1, self.order + 1):
                    indices['diagonal'].append((i, k, m))

        # Cross-memory terms (single-input only for now)
        if not self.is_diagonal and I == 1:
            for k1 in range(self.memory_length):
                for k2 in range(self.memory_length):
                    lag_dist = abs(k1 - k2)
                    if lag_dist == 0 or lag_dist > self.config.max_cross_lag_distance:
                        continue

                    # Check lead/lag
                    if k1 < k2 and not self.config.include_lead_terms:
                        continue
                    if k1 > k2 and not self.config.include_lag_terms:
                        continue

                    # Add valid order combinations
                    for p in range(1, self.order + 1):
                        for q in range(1, self.order + 1):
                            if p + q <= self.config.max_cross_order:
                                # (k1, order_k1, k2, order_k2)
                                indices['cross_terms'].append((k1, p, k2, q))

        return indices

    def _build_design_matrix(self, x: np.ndarray) -> np.ndarray:
        """Build design matrix for GMP regression.

        Args:
            x: Input data, shape (T, I) for MIMO or (T,) for SISO.

        Returns:
            Design matrix Phi, shape (T, P) where P is total number of terms.
        """
        if x.ndim == 1:
            x = x[:, None]  # (T, 1)

        T, I = x.shape
        P = self.total_terms * I if I > 1 else self.total_terms

        Phi = np.zeros((T, P))
        col_idx = 0

        # Build delay matrices for each input
        delay_matrices = []
        for i in range(I):
            X_delay = build_mimo_delay_matrix(x[:, i], self.memory_length)
            # Pad to match T
            X_delay_padded = np.vstack([
                np.zeros((self.memory_length - 1, self.memory_length)),
                X_delay
            ])
            delay_matrices.append(X_delay_padded)

        # Generate term indices
        term_indices = self._generate_term_indices(I)

        # Diagonal terms
        for (i, k, m) in term_indices['diagonal']:
            Phi[:, col_idx] = delay_matrices[i][:, k] ** m
            col_idx += 1

        # Cross-memory terms (SISO only for now)
        if not self.is_diagonal and I == 1:
            for (k1, p, k2, q) in term_indices['cross_terms']:
                Phi[:, col_idx] = (delay_matrices[0][:, k1] ** p) * (delay_matrices[0][:, k2] ** q)
                col_idx += 1

        return Phi

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'GeneralizedMemoryPolynomial':
        """Fit GMP model to input-output data using regularized least squares.

        Args:
            x: Input data, shape (T, I) for MIMO or (T,) for SISO.
            y: Output data, shape (T, O) for multi-output or (T,) for single output.

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If data is invalid.
        """
        # Validate and canonicalize data
        validate_mimo_data(x, y)
        x_canon = canonicalize_input(x)  # (T, I)
        y_canon = canonicalize_output(y)  # (T, O)

        T, I = x_canon.shape
        O = y_canon.shape[1]

        self._n_inputs = I
        self._n_outputs = O

        if self.config.verbose:
            print(f"Fitting GMP: I={I}, O={O}, N={self.memory_length}, M={self.order}")
            print(f"Total terms: {self.total_terms}")

        # Build design matrix
        Phi = self._build_design_matrix(x_canon if I > 1 else x_canon[:, 0])

        # Fit separate model for each output
        self._coefficients = []
        for o in range(O):
            y_o = y_canon[:, o]

            # Regularized least squares: (Phi^T Phi + lambda I) w = Phi^T y
            PhiT_Phi = Phi.T @ Phi
            PhiT_y = Phi.T @ y_o

            # Add regularization
            reg_matrix = self.config.regularization * np.eye(Phi.shape[1])
            A = PhiT_Phi + reg_matrix

            # Solve
            w = np.linalg.solve(A, PhiT_y)
            self._coefficients.append(w)

            if self.config.verbose:
                y_pred = Phi @ w
                nmse = np.mean((y_o - y_pred) ** 2) / np.var(y_o)
                print(f"Output {o}: NMSE = {nmse:.6e}")

        # Store term indices for export
        self._term_indices = self._generate_term_indices(I)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict output for new input using fitted model.

        Args:
            x: Input data, shape (T, I) for MIMO or (T,) for SISO.

        Returns:
            Predicted output, shape (T, O) for multi-output or (T,) for single output.

        Raises:
            RuntimeError: If model not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Canonicalize input
        x_canon = canonicalize_input(x)  # (T, I)

        # Build design matrix
        Phi = self._build_design_matrix(x_canon if self._n_inputs > 1 else x_canon[:, 0])

        # Predict for each output
        predictions = []
        for o in range(self._n_outputs):
            y_pred = Phi @ self._coefficients[o]
            predictions.append(y_pred)

        # Stack and return
        y_pred_all = np.column_stack(predictions)

        # Return as (T,) for single output
        if self._n_outputs == 1:
            return y_pred_all[:, 0]
        return y_pred_all

    def get_coefficients(self, output_idx: int = 0) -> Dict[str, np.ndarray]:
        """Get model coefficients organized by term type.

        Args:
            output_idx: Output channel index (0-based).

        Returns:
            Dictionary with keys:
                - 'diagonal': Diagonal term coefficients, shape (I*N*M,)
                - 'cross_terms': Cross-term coefficients (if enabled)

        Raises:
            RuntimeError: If model not fitted.
            IndexError: If output_idx is invalid.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting coefficients")
        if output_idx < 0 or output_idx >= self._n_outputs:
            raise IndexError(f"output_idx {output_idx} out of range [0, {self._n_outputs})")

        w = self._coefficients[output_idx]

        # Split coefficients by term type
        n_diagonal = len(self._term_indices['diagonal'])
        n_cross = len(self._term_indices['cross_terms'])

        result = {
            'diagonal': w[:n_diagonal],
        }

        if n_cross > 0:
            result['cross_terms'] = w[n_diagonal:n_diagonal + n_cross]

        return result

    def export_model(self, output_idx: int = 0) -> Dict:
        """Export model for C++/Rust deployment.

        Args:
            output_idx: Output channel index (0-based).

        Returns:
            Dictionary with complete model description:
                - model_type: 'GMP'
                - memory_length: N
                - order: M
                - n_inputs: I
                - coefficients: organized by term type
                - config: GMP configuration
                - term_indices: description of each term

        Raises:
            RuntimeError: If model not fitted.
            IndexError: If output_idx is invalid.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before export")
        if output_idx < 0 or output_idx >= self._n_outputs:
            raise IndexError(f"output_idx {output_idx} out of range [0, {self._n_outputs})")

        return {
            'model_type': 'GMP',
            'memory_length': self.memory_length,
            'order': self.order,
            'n_inputs': self._n_inputs,
            'n_outputs': self._n_outputs,
            'coefficients': self.get_coefficients(output_idx),
            'config': {
                'max_cross_lag_distance': self.config.max_cross_lag_distance,
                'max_cross_order': self.config.max_cross_order,
                'include_lead_terms': self.config.include_lead_terms,
                'include_lag_terms': self.config.include_lag_terms,
                'regularization': self.config.regularization,
            },
            'term_indices': {
                'diagonal': self._term_indices['diagonal'],
                'cross_terms': self._term_indices['cross_terms'],
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted=True" if self.is_fitted else "fitted=False"
        return (
            f"GeneralizedMemoryPolynomial(memory_length={self.memory_length}, "
            f"order={self.order}, total_terms={self.total_terms}, {fitted_str})"
        )
