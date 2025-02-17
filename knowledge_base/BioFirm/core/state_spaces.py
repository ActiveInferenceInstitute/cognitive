"""
Core state space implementations for BioFirm framework.
Defines the fundamental state representations and transformations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import numpy as np
from abc import ABC, abstractmethod

@dataclass
class StateSpace:
    """Abstract representation of state spaces in active inference models."""
    dimensions: List[int]
    labels: Dict[str, List[str]]
    mappings: Dict[str, np.ndarray]
    hierarchical_levels: Optional[int] = 1
    scale: Optional[str] = None  # spatial scale of the state space
    temporal_resolution: Optional[str] = None  # temporal resolution

    def __post_init__(self):
        """Validate state space configuration."""
        self._validate_dimensions()
        self._validate_mappings()
        
    def _validate_dimensions(self):
        """Ensure dimensions are properly specified."""
        if not all(isinstance(d, int) and d > 0 for d in self.dimensions):
            raise ValueError("All dimensions must be positive integers")
            
    def _validate_mappings(self):
        """Validate mapping matrices."""
        for key, mapping in self.mappings.items():
            if not isinstance(mapping, np.ndarray):
                raise TypeError(f"Mapping {key} must be a numpy array")

@dataclass
class BioregionalState:
    """Comprehensive bioregional state representation."""
    ecological_state: Dict[str, float] = field(default_factory=lambda: {
        "biodiversity": 0.0,
        "habitat_connectivity": 0.0,
        "ecosystem_services": 0.0,
        "species_richness": 0.0,
        "ecological_integrity": 0.0
    })
    
    climate_state: Dict[str, float] = field(default_factory=lambda: {
        "temperature": 0.0,
        "precipitation": 0.0,
        "carbon_storage": 0.0,
        "albedo": 0.0,
        "extreme_events": 0.0
    })
    
    social_state: Dict[str, float] = field(default_factory=lambda: {
        "community_engagement": 0.0,
        "traditional_knowledge": 0.0,
        "stewardship_practices": 0.0,
        "resource_governance": 0.0,
        "social_resilience": 0.0
    })
    
    economic_state: Dict[str, float] = field(default_factory=lambda: {
        "sustainable_livelihoods": 0.0,
        "circular_economy": 0.0,
        "ecosystem_valuation": 0.0,
        "green_infrastructure": 0.0,
        "resource_efficiency": 0.0
    })

    def to_vector(self) -> np.ndarray:
        """Convert state to vector representation."""
        return np.array([
            *list(self.ecological_state.values()),
            *list(self.climate_state.values()),
            *list(self.social_state.values()),
            *list(self.economic_state.values())
        ])
    
    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'BioregionalState':
        """Create state from vector representation."""
        if len(vector) != 20:  # Total number of state variables
            raise ValueError("Vector must have length 20")
            
        state = cls()
        
        # Update ecological state
        for i, key in enumerate(state.ecological_state.keys()):
            state.ecological_state[key] = vector[i]
            
        # Update climate state
        offset = len(state.ecological_state)
        for i, key in enumerate(state.climate_state.keys()):
            state.climate_state[key] = vector[offset + i]
            
        # Update social state
        offset += len(state.climate_state)
        for i, key in enumerate(state.social_state.keys()):
            state.social_state[key] = vector[offset + i]
            
        # Update economic state
        offset += len(state.social_state)
        for i, key in enumerate(state.economic_state.keys()):
            state.economic_state[key] = vector[offset + i]
            
        return state

    def validate_bounds(self) -> bool:
        """Validate that all state variables are within [0,1]."""
        for state_dict in [self.ecological_state, self.climate_state, 
                          self.social_state, self.economic_state]:
            if not all(0 <= v <= 1 for v in state_dict.values()):
                return False
        return True

class StateTransformation(ABC):
    """Abstract base class for state transformations."""
    
    @abstractmethod
    def transform(self, state: BioregionalState) -> BioregionalState:
        """Transform state according to implementation."""
        pass
    
    @abstractmethod
    def inverse_transform(self, state: BioregionalState) -> BioregionalState:
        """Inverse transform state according to implementation."""
        pass

class ScaleTransformation(StateTransformation):
    """Transform states between different spatial scales."""
    
    def __init__(self, source_scale: str, target_scale: str,
                 aggregation_weights: Optional[Dict[str, np.ndarray]] = None):
        self.source_scale = source_scale
        self.target_scale = target_scale
        self.aggregation_weights = aggregation_weights or self._default_weights()
        
    def transform(self, state: BioregionalState) -> BioregionalState:
        """Transform state from source to target scale."""
        if self.source_scale == self.target_scale:
            return state
            
        vector = state.to_vector()
        transformed = np.dot(self.aggregation_weights[self.target_scale], vector)
        return BioregionalState.from_vector(transformed)
        
    def inverse_transform(self, state: BioregionalState) -> BioregionalState:
        """Transform state from target back to source scale."""
        if self.source_scale == self.target_scale:
            return state
            
        vector = state.to_vector()
        transformed = np.dot(
            np.linalg.pinv(self.aggregation_weights[self.target_scale]),
            vector
        )
        return BioregionalState.from_vector(transformed)
        
    def _default_weights(self) -> Dict[str, np.ndarray]:
        """Create default aggregation weights."""
        n_dims = 20  # Total state dimensions
        scales = ["local", "landscape", "regional", "bioregional"]
        
        weights = {}
        for scale in scales:
            # Create identity matrix as default
            weights[scale] = np.eye(n_dims)
            
        return weights 