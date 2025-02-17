"""
BioFirm framework for Earth System Active Inference.
"""

from .earth_systems import (
    SystemState, EcologicalState, ClimateState, HumanImpactState, SimulationConfig
)
from .simulator import EarthSystemSimulator

__all__ = [
    'SystemState',
    'EcologicalState',
    'ClimateState',
    'HumanImpactState',
    'SimulationConfig',
    'EarthSystemSimulator'
] 