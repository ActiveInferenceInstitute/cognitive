"""
Active Inference Framework for Cognitive Modeling.
"""

from .models.active_inference import (
    ActiveInferenceDispatcher,
    ActiveInferenceFactory,
    ActiveInferenceModel,
    AdaptiveControl,
    HomeostaticControl,
    HomeostaticFactory,
    HomeostaticInference,
    InferenceConfig,
    InferenceMethod,
    ModelState,
    PolicyType,
)
from .utils.matrix_utils import (
    compute_entropy,
    ensure_matrix_properties,
    expected_free_energy,
    kl_divergence,
    softmax,
)
from .visualization.matrix_plots import MatrixPlotter

__version__ = "0.1.0"

__all__ = [
    "ActiveInferenceDispatcher",
    "ActiveInferenceFactory",
    "ActiveInferenceModel",
    "AdaptiveControl",
    "HomeostaticControl",
    "HomeostaticFactory",
    "HomeostaticInference",
    "InferenceConfig",
    "InferenceMethod",
    "MatrixPlotter",
    "ModelState",
    "PolicyType",
    "compute_entropy",
    "ensure_matrix_properties",
    "expected_free_energy",
    "kl_divergence",
    "softmax",
]
