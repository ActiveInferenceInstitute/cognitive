"""
Transition model implementation for BioFirm framework.
Handles state evolution and action effects.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import multivariate_normal

from .state_spaces import StateSpace, BioregionalState

@dataclass
class TransitionModel:
    """Dynamic transition model for state evolution."""
    state_space: StateSpace
    action_space: StateSpace
    transition_matrices: Dict[str, np.ndarray]  # B matrices
    temporal_horizon: int
    control_modes: List[str] = ["homeostatic", "goal_directed", "exploratory"]
    
    def __post_init__(self):
        """Validate transition model configuration."""
        self._validate_spaces()
        self._validate_matrices()
        self._setup_control_modes()
        
    def _validate_spaces(self):
        """Ensure state and action spaces are compatible."""
        if not isinstance(self.state_space, StateSpace):
            raise TypeError("state_space must be a StateSpace instance")
        if not isinstance(self.action_space, StateSpace):
            raise TypeError("action_space must be a StateSpace instance")
            
    def _validate_matrices(self):
        """Validate transition matrices."""
        state_dim = np.prod(self.state_space.dimensions)
        for action, matrix in self.transition_matrices.items():
            if matrix.shape != (state_dim, state_dim):
                raise ValueError(
                    f"Transition matrix for action {action} has invalid shape "
                    f"{matrix.shape}, expected ({state_dim}, {state_dim})"
                )
                
    def _setup_control_modes(self):
        """Configure control modes."""
        if not all(mode in self.VALID_MODES for mode in self.control_modes):
            raise ValueError(f"Invalid control mode in {self.control_modes}")
            
    VALID_MODES = {
        "homeostatic": lambda s, g: -np.sum((s - g) ** 2),
        "goal_directed": lambda s, g: -np.sum(np.abs(s - g)),
        "exploratory": lambda s, _: -np.sum(s * np.log(s + 1e-10))
    }
    
    def predict_next_state(self,
                          current_state: BioregionalState,
                          action: str,
                          noise_scale: float = 0.1) -> BioregionalState:
        """Predict next state given current state and action."""
        # Get transition matrix for action
        if action not in self.transition_matrices:
            raise ValueError(f"Invalid action: {action}")
            
        B = self.transition_matrices[action]
        
        # Convert state to vector
        state_vector = current_state.to_vector()
        
        # Apply transition
        next_state_vector = np.dot(B, state_vector)
        
        # Add noise
        noise = np.random.normal(0, noise_scale, size=len(next_state_vector))
        next_state_vector += noise
        
        # Ensure bounds
        next_state_vector = np.clip(next_state_vector, 0, 1)
        
        # Convert back to BioregionalState
        return BioregionalState.from_vector(next_state_vector)
        
    def evaluate_action(self,
                       current_state: BioregionalState,
                       goal_state: BioregionalState,
                       action: str,
                       mode: str = "homeostatic") -> float:
        """Evaluate action using specified control mode."""
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid control mode: {mode}")
            
        # Predict next state
        predicted_next = self.predict_next_state(current_state, action)
        
        # Evaluate using control mode
        return self.VALID_MODES[mode](
            predicted_next.to_vector(),
            goal_state.to_vector()
        )
        
    def get_optimal_action(self,
                          current_state: BioregionalState,
                          goal_state: BioregionalState,
                          mode: str = "homeostatic") -> str:
        """Get optimal action using specified control mode."""
        action_values = {
            action: self.evaluate_action(current_state, goal_state, action, mode)
            for action in self.transition_matrices.keys()
        }
        return max(action_values.items(), key=lambda x: x[1])[0]

class HierarchicalTransition:
    """Hierarchical transition model for multi-scale dynamics."""
    
    def __init__(self,
                 models: Dict[str, TransitionModel],
                 scale_couplings: Dict[Tuple[str, str], float]):
        """
        Initialize hierarchical transition model.
        
        Args:
            models: Dictionary mapping scales to TransitionModels
            scale_couplings: Dictionary mapping scale pairs to coupling strengths
        """
        self.models = models
        self.scale_couplings = scale_couplings
        self._validate_configuration()
        
    def _validate_configuration(self):
        """Validate hierarchical model configuration."""
        # Validate models
        if not all(isinstance(m, TransitionModel) for m in self.models.values()):
            raise TypeError("All models must be TransitionModel instances")
            
        # Validate couplings
        scales = set(self.models.keys())
        for (scale1, scale2) in self.scale_couplings.keys():
            if scale1 not in scales or scale2 not in scales:
                raise ValueError(f"Invalid scales in coupling: {scale1}, {scale2}")
                
    def predict_hierarchy(self,
                         current_states: Dict[str, BioregionalState],
                         actions: Dict[str, str]) -> Dict[str, BioregionalState]:
        """Predict next states across hierarchy."""
        # First pass: independent predictions
        next_states = {
            scale: self.models[scale].predict_next_state(
                current_states[scale], actions[scale]
            )
            for scale in self.models.keys()
        }
        
        # Second pass: apply cross-scale coupling
        coupled_states = next_states.copy()
        for (scale1, scale2), coupling in self.scale_couplings.items():
            # Convert states to vectors for coupling
            v1 = next_states[scale1].to_vector()
            v2 = next_states[scale2].to_vector()
            
            # Apply coupling
            coupled_v1 = (1 - coupling) * v1 + coupling * v2
            
            # Convert back to BioregionalState
            coupled_states[scale1] = BioregionalState.from_vector(coupled_v1)
            
        return coupled_states

class EcologicalConstraints:
    """Enforces ecological constraints on transitions."""
    
    def __init__(self, constraints: Dict[str, Dict[str, float]]):
        """
        Initialize constraints.
        
        Args:
            constraints: Dictionary mapping variable pairs to min/max ratios
        """
        self.constraints = constraints
        
    def apply_constraints(self, state: BioregionalState) -> BioregionalState:
        """Apply ecological constraints to state."""
        # Convert to modifiable form
        state_dict = {
            "ecological": state.ecological_state.copy(),
            "climate": state.climate_state.copy(),
            "social": state.social_state.copy(),
            "economic": state.economic_state.copy()
        }
        
        # Apply each constraint
        for (var1, var2), limits in self.constraints.items():
            domain1, var1_name = var1.split(".")
            domain2, var2_name = var2.split(".")
            
            value1 = state_dict[domain1][var1_name]
            value2 = state_dict[domain2][var2_name]
            
            ratio = value1 / (value2 + 1e-10)
            
            if ratio < limits["min"]:
                # Increase value1 or decrease value2
                adjustment = limits["min"] * value2 - value1
                state_dict[domain1][var1_name] += adjustment
            elif ratio > limits["max"]:
                # Decrease value1 or increase value2
                adjustment = value1 - limits["max"] * value2
                state_dict[domain1][var1_name] -= adjustment
                
        # Create new constrained state
        return BioregionalState(
            ecological_state=state_dict["ecological"],
            climate_state=state_dict["climate"],
            social_state=state_dict["social"],
            economic_state=state_dict["economic"]
        ) 