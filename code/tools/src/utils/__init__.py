"""
Utility functions for Active Inference models.
"""

from .matrix_utils import (
    compute_entropy,
    ensure_matrix_properties,
    expected_free_energy,
    kl_divergence,
    softmax,
)

__all__ = [
    "compute_entropy",
    "ensure_matrix_properties",
    "expected_free_energy",
    "kl_divergence",
    "softmax",
]
