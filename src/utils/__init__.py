"""
Utility functions for Active Inference models.
"""

from .matrix_utils import (
    ensure_matrix_properties,
    compute_entropy,
    softmax,
    kl_divergence,
    expected_free_energy
) 