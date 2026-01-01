"""
Utility functions for Volterra processing.

This module provides helper functions for:
- MIMO shape validation and canonicalization
- Data type handling
- Parameter validation
"""

from volterra.utils.shapes import (
    canonicalize_input,
    canonicalize_output,
    validate_mimo_data,
    infer_dimensions,
)

__all__ = [
    "canonicalize_input",
    "canonicalize_output",
    "validate_mimo_data",
    "infer_dimensions",
]
