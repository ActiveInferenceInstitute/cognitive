"""
Active Inference Framework for Cognitive Modeling.
"""

from .models.active_inference.base import ActiveInferenceModel
from .utils.matrix_utils import (
    ensure_matrix_properties,
    compute_entropy,
    softmax,
    kl_divergence,
    expected_free_energy
)
from .visualization.matrix_plots import MatrixPlotter

__version__ = '0.1.0' 