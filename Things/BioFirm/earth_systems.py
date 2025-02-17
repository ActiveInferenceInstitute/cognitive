"""
Multi-scale Earth Systems Active Inference implementation.
Provides nested control and inference for earth system stewardship.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import yaml
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Add BioFirm directory to path
BIOFIRM_DIR = Path(__file__).parent
if str(BIOFIRM_DIR) not in sys.path:
    sys.path.append(str(BIOFIRM_DIR))

try:
    from homeostatic import (
        StateSpace, HomeostaticInference, ControlMode,
        HomestaticControl, AdaptiveControl
    )
    from active_inference.dispatcher import ActiveInferenceDispatcher
except ImportError as e:
    logger.error(f"Error importing local modules: {str(e)}")
    raise

@dataclass
class EcologicalState:
    """Multi-scale ecological state representation."""
    biodiversity: Dict[str, float]
    biomass: Dict[str, float]
    soil_health: Dict[str, float]
    water_cycles: Dict[str, float]
    energy_flows: Dict[str, float]
    resilience: Dict[str, float]

@dataclass
class ClimateState:
    """Climate system state representation."""
    temperature: Dict[str, float]
    precipitation: Dict[str, float]
    wind_patterns: Dict[str, float]
    carbon_cycles: Dict[str, float]
    energy_balance: Dict[str, float]

@dataclass
class HumanImpactState:
    """Human activity impact representation."""
    land_use: Dict[str, float]
    resource_extraction: Dict[str, float]
    pollution_levels: Dict[str, float]
    restoration_efforts: Dict[str, float]
    social_indicators: Dict[str, float]

@dataclass
class SystemState:
    """Combined system state."""
    ecological: EcologicalState
    climate: ClimateState
    human: HumanImpactState
    timestamp: float
    scale: str

@dataclass
class ScaleConfig:
    """Configuration for a single scale."""
    scale: float
    processes: List[str]
    coupling_strength: float

@dataclass
class SpatialConfig:
    """Configuration for spatial hierarchy."""
    scale: float
    entities: List[str]
    interaction_range: float

@dataclass
class InterventionConfig:
    """Configuration for intervention strategies."""
    implementation_rate: float
    monitoring_frequency: int
    weight: float
    restoration_targets: Optional[List[str]] = None
    intervention_types: Optional[List[str]] = None
    intervention_areas: Optional[List[str]] = None
    risk_tolerance: Optional[float] = None
    uncertainty_threshold: Optional[float] = None
    community_threshold: Optional[float] = None

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    plot_types: List[str] = field(default_factory=lambda: ["line", "heatmap", "network", "scatter"])
    update_frequency: int = 10
    save_format: str = "png"
    dpi: int = 300
    style: str = "seaborn-whitegrid"
    color_scheme: str = "viridis"
    plot_dimensions: Dict[str, int] = field(default_factory=lambda: {"width": 12, "height": 8})

@dataclass
class SimulationConfig:
    """Configuration for earth system simulation."""
    duration: float
    time_step: float
    spatial_resolution: Dict[str, float]
    noise_levels: Dict[str, float]
    intervention_frequency: int
    random_seed: Optional[int] = None
    temporal_hierarchy: Dict[str, ScaleConfig] = field(default_factory=dict)
    spatial_hierarchy: Dict[str, SpatialConfig] = field(default_factory=dict)
    intervention_strategies: Dict[str, InterventionConfig] = field(default_factory=dict)
    visualization: Optional[VisualizationConfig] = None

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SimulationConfig':
        """Create SimulationConfig from dictionary."""
        sim_config = config_dict.get('simulation', {})
        temporal = {k: ScaleConfig(**v) for k, v in config_dict.get('temporal_hierarchy', {}).items()}
        spatial = {k: SpatialConfig(**v) for k, v in config_dict.get('spatial_hierarchy', {}).items()}
        interventions = {k: InterventionConfig(**v) for k, v in config_dict.get('intervention_strategies', {}).items()}
        viz_config = VisualizationConfig(**config_dict['visualization']) if 'visualization' in config_dict else None
        
        return cls(
            duration=sim_config.get('duration', 1000.0),
            time_step=sim_config.get('time_step', 0.1),
            spatial_resolution=sim_config.get('spatial_resolution', {}),
            noise_levels=sim_config.get('noise_levels', {}),
            intervention_frequency=sim_config.get('intervention_frequency', 10),
            random_seed=sim_config.get('random_seed'),
            temporal_hierarchy=temporal,
            spatial_hierarchy=spatial,
            intervention_strategies=interventions,
            visualization=viz_config
        )

# Export classes
__all__ = [
    'SystemState',
    'EcologicalState',
    'ClimateState',
    'HumanImpactState',
    'SimulationConfig'
] 