"""
Active Inference implementation for BioFirm framework.
"""

from .dispatcher import (
    ActiveInferenceDispatcher,
    ActiveInferenceFactory,
    InferenceConfig,
    InferenceMethod,
    PolicyType,
    ModelState
)

__all__ = [
    'ActiveInferenceDispatcher',
    'ActiveInferenceFactory',
    'InferenceConfig',
    'InferenceMethod',
    'PolicyType',
    'ModelState'
] 