"""
Homeostatic control implementation using Active Inference.
Provides abstractions and implementations for homeostatic control systems.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path

from .base import ActiveInferenceModel, ModelState
from .dispatcher import ActiveInferenceDispatcher, InferenceConfig
from ..matrices.matrix_ops import MatrixOps

@dataclass
class StateSpace:
    """Abstract representation of state spaces in active inference models."""
    dimensions: List[int]
    labels: Dict[str, List[str]]
    mappings: Dict[str, np.ndarray]
    hierarchical_levels: Optional[int] = 1
    
    def validate(self) -> bool:
        """Validate state space configuration."""
        # Check dimensions match labels
        for dim, label_list in zip(self.dimensions, self.labels.values()):
            if len(label_list) != dim:
                return False
        # Check mappings are valid
        for mapping in self.mappings.values():
            if not isinstance(mapping, np.ndarray):
                return False
        return True

@dataclass
class ObservationModel:
    """Generalized observation model for active inference."""
    state_space: StateSpace
    observation_space: StateSpace
    likelihood_matrix: np.ndarray
    noise_model: str = "gaussian"
    precision: float = 1.0
    
    def compute_likelihood(self,
                         observation: np.ndarray,
                         state: np.ndarray) -> float:
        """Compute observation likelihood given state."""
        if self.noise_model == "gaussian":
            prediction = np.dot(self.likelihood_matrix, state)
            return np.exp(-0.5 * self.precision * np.sum(np.square(observation - prediction)))
        else:
            raise ValueError(f"Unsupported noise model: {self.noise_model}")

@dataclass
class TransitionModel:
    """Dynamic transition model for state evolution."""
    state_space: StateSpace
    action_space: StateSpace
    transition_matrices: Dict[str, np.ndarray]
    temporal_horizon: int
    control_modes: List[str] = field(default_factory=lambda: ["homeostatic"])
    
    def get_transition_matrix(self, action: Union[int, str]) -> np.ndarray:
        """Get transition matrix for given action."""
        if isinstance(action, int):
            action = self.action_space.labels["actions"][action]
        return self.transition_matrices[action]

class ControlMode(ABC):
    """Abstract base class for control modes."""
    
    @abstractmethod
    def compute_policy_prior(self,
                           state: ModelState,
                           goal: np.ndarray) -> np.ndarray:
        """Compute policy prior based on control mode."""
        pass

class HomestaticControl(ControlMode):
    """Homeostatic control mode implementation."""
    
    def __init__(self,
                 bounds: Tuple[float, float],
                 target_state: Union[str, int],
                 weight: float = 1.0):
        self.bounds = bounds
        self.target_state = target_state
        self.weight = weight
    
    def compute_policy_prior(self,
                           state: ModelState,
                           goal: np.ndarray) -> np.ndarray:
        """Compute homeostatic control policy prior."""
        deviation = np.abs(state.beliefs - goal)
        return np.exp(-self.weight * deviation)

class AdaptiveControl(ControlMode):
    """Adaptive control mode implementation."""
    
    def __init__(self,
                 learning_rate: float = 0.1,
                 exploration_weight: float = 0.3):
        self.learning_rate = learning_rate
        self.exploration_weight = exploration_weight
    
    def compute_policy_prior(self,
                           state: ModelState,
                           goal: np.ndarray) -> np.ndarray:
        """Compute adaptive control policy prior."""
        # Balance exploitation and exploration
        exploitation = -np.abs(state.beliefs - goal)
        exploration = -state.prediction_error * np.ones_like(state.beliefs)
        return np.exp(
            (1 - self.exploration_weight) * exploitation +
            self.exploration_weight * exploration
        )

class HomeostaticInference(ActiveInferenceModel):
    """Homeostatic control using Active Inference."""
    
    def __init__(self,
                 config_path: Union[str, Path],
                 control_mode: ControlMode):
        super().__init__(config_path)
        self.control_mode = control_mode
        self._setup_spaces()
        
    def _setup_spaces(self):
        """Setup state spaces from configuration."""
        config = self._load_config(self.config_path)
        
        # Setup state spaces
        self.state_space = StateSpace(
            dimensions=config["state_spaces"]["environment"]["dimensions"],
            labels=config["state_spaces"]["environment"]["labels"],
            mappings=config["state_spaces"]["environment"]["mappings"]
        )
        
        self.observation_space = StateSpace(
            dimensions=config["state_spaces"]["observation"]["dimensions"],
            labels=config["state_spaces"]["observation"]["labels"],
            mappings=config["state_spaces"]["observation"]["mappings"]
        )
        
        self.action_space = StateSpace(
            dimensions=config["state_spaces"]["action"]["dimensions"],
            labels=config["state_spaces"]["action"]["labels"],
            mappings=config["state_spaces"]["action"]["mappings"]
        )
        
    def update_beliefs(self, observation: np.ndarray) -> np.ndarray:
        """Update beliefs using active inference."""
        # Get policy prior from control mode
        policy_prior = self.control_mode.compute_policy_prior(
            self.state,
            self.config["target_state"]
        )
        
        # Update beliefs using dispatcher
        return self.dispatcher.dispatch_belief_update(
            observation=observation,
            current_state=self.state,
            policy_prior=policy_prior
        )
    
    def select_action(self) -> int:
        """Select action using active inference."""
        policies = self.dispatcher.dispatch_policy_inference(
            state=self.state,
            goal_prior=self.control_mode.compute_policy_prior(
                self.state,
                self.config["target_state"]
            )
        )
        return np.argmax(policies)
    
    def update_parameters(self, performance: Dict[str, float]):
        """Update model parameters based on performance."""
        if isinstance(self.control_mode, AdaptiveControl):
            # Update exploration weight based on performance
            stability = performance.get("stability_index", 0.0)
            self.control_mode.exploration_weight = np.clip(
                0.5 - 0.4 * stability,  # Reduce exploration as stability increases
                0.1,  # Minimum exploration
                0.5   # Maximum exploration
            )

class HomeostaticFactory:
    """Factory for creating homeostatic control instances."""
    
    @staticmethod
    def create_basic(config_path: Union[str, Path]) -> HomeostaticInference:
        """Create basic homeostatic control instance."""
        control_mode = HomestaticControl(
            bounds=(-1.0, 1.0),
            target_state="MEDIUM",
            weight=1.0
        )
        return HomeostaticInference(config_path, control_mode)
    
    @staticmethod
    def create_adaptive(config_path: Union[str, Path]) -> HomeostaticInference:
        """Create adaptive homeostatic control instance."""
        control_mode = AdaptiveControl(
            learning_rate=0.1,
            exploration_weight=0.3
        )
        return HomeostaticInference(config_path, control_mode) 