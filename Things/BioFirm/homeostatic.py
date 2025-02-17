"""
Homeostatic control implementation for BioFirm framework.
"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import yaml
import logging

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class StateSpace:
    """Abstract representation of state spaces in active inference models."""
    dimensions: List[int]
    labels: Dict[str, List[str]]
    mappings: Dict[str, np.ndarray]
    hierarchical_levels: Optional[int] = 1
    
    def validate(self) -> bool:
        """Validate state space configuration."""
        try:
            # Check dimensions match labels
            for dim, label_list in zip(self.dimensions, self.labels.values()):
                if len(label_list) != dim:
                    return False
            # Check mappings are valid
            for mapping in self.mappings.values():
                if not isinstance(mapping, np.ndarray):
                    return False
            return True
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False

@dataclass
class ModelState:
    """Represents the current state of an Active Inference model."""
    beliefs: np.ndarray
    policies: np.ndarray
    precision: float
    free_energy: float
    prediction_error: float

    def validate(self) -> bool:
        """Validate model state."""
        try:
            if not isinstance(self.beliefs, np.ndarray) or not isinstance(self.policies, np.ndarray):
                return False
            if not isinstance(self.precision, float) or not isinstance(self.free_energy, float):
                return False
            if not isinstance(self.prediction_error, float):
                return False
            return True
        except Exception as e:
            logger.error(f"State validation error: {str(e)}")
            return False

class ControlMode(ABC):
    """Abstract base class for control modes."""
    
    @abstractmethod
    def compute_policy_prior(self,
                           state: ModelState,
                           goal: np.ndarray) -> np.ndarray:
        """Compute policy prior based on control mode."""
        pass

    def validate_inputs(self, state: ModelState, goal: np.ndarray) -> bool:
        """Validate inputs for policy computation."""
        try:
            if not isinstance(state, ModelState) or not state.validate():
                return False
            if not isinstance(goal, np.ndarray):
                return False
            return True
        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            return False

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
        if not self.validate_inputs(state, goal):
            raise ValueError("Invalid inputs for policy computation")
        
        try:
            deviation = np.abs(state.beliefs - goal)
            return np.exp(-self.weight * deviation)
        except Exception as e:
            logger.error(f"Error computing policy prior: {str(e)}")
            raise

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
        if not self.validate_inputs(state, goal):
            raise ValueError("Invalid inputs for policy computation")
        
        try:
            # Balance exploitation and exploration
            exploitation = -np.abs(state.beliefs - goal)
            exploration = -state.prediction_error * np.ones_like(state.beliefs)
            return np.exp(
                (1 - self.exploration_weight) * exploitation +
                self.exploration_weight * exploration
            )
        except Exception as e:
            logger.error(f"Error computing adaptive policy prior: {str(e)}")
            raise

class HomeostaticInference:
    """Homeostatic control using Active Inference."""
    
    def __init__(self,
                 config_path: Union[str, Path],
                 control_mode: ControlMode):
        """Initialize homeostatic inference.
        
        Args:
            config_path: Path to configuration file
            control_mode: Control mode instance
        """
        self.config_path = Path(config_path)
        self.control_mode = control_mode
        self.config = self._load_config()
        self._initialize_matrices()
        self.state = self._initialize_state()
    
    def _load_config(self) -> Dict:
        """Load model configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self._validate_config(config)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
    
    def _validate_config(self, config: Dict) -> bool:
        """Validate configuration structure."""
        required_fields = [
            'observation_model',
            'transition_model',
            'preference_model',
            'prior_beliefs'
        ]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
        return True
    
    def _initialize_matrices(self):
        """Initialize model matrices."""
        try:
            # Initialize observation model (A matrix)
            self.A = np.array(self.config.get('observation_model', np.eye(5)))
            
            # Initialize transition model (B matrix)
            self.B = np.array(self.config.get('transition_model', np.eye(5)))
            
            # Initialize preference model (C matrix)
            self.C = np.array(self.config.get('preference_model', np.zeros(5)))
            
            # Initialize prior beliefs (D matrix)
            self.D = np.array(self.config.get('prior_beliefs', np.ones(5) / 5))
            
            self._validate_matrices()
        except Exception as e:
            logger.error(f"Error initializing matrices: {str(e)}")
            raise
    
    def _validate_matrices(self):
        """Validate matrix dimensions and properties."""
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A matrix must be square")
        if self.B.shape[0] != self.B.shape[1]:
            raise ValueError("B matrix must be square")
        if len(self.C) != self.A.shape[0]:
            raise ValueError("C vector dimension mismatch")
        if len(self.D) != self.A.shape[0]:
            raise ValueError("D vector dimension mismatch")
    
    def _initialize_state(self) -> ModelState:
        """Initialize model state."""
        try:
            state = ModelState(
                beliefs=self.D.copy(),
                policies=np.ones(len(self.D)) / len(self.D),
                precision=1.0,
                free_energy=0.0,
                prediction_error=0.0
            )
            if not state.validate():
                raise ValueError("Invalid initial state")
            return state
        except Exception as e:
            logger.error(f"Error initializing state: {str(e)}")
            raise
    
    def update_beliefs(self, observation: np.ndarray) -> np.ndarray:
        """Update beliefs using active inference."""
        try:
            # Generate prediction
            prediction = np.dot(self.A, self.state.beliefs)
            
            # Compute prediction error
            prediction_error = observation - prediction
            
            # Update beliefs using precision-weighted prediction errors
            belief_update = self.state.precision * prediction_error
            self.state.beliefs += belief_update
            
            # Normalize beliefs
            self.state.beliefs = self.state.beliefs / np.sum(self.state.beliefs)
            
            # Update prediction error
            self.state.prediction_error = np.mean(np.square(prediction_error))
            
            return self.state.beliefs
        except Exception as e:
            logger.error(f"Error updating beliefs: {str(e)}")
            raise
    
    def select_action(self) -> int:
        """Select action using active inference."""
        try:
            # Compute expected free energy for each policy
            expected_free_energy = self._compute_expected_free_energy()
            
            # Get policy prior from control mode
            policy_prior = self.control_mode.compute_policy_prior(
                self.state,
                self.config["target_state"]
            )
            
            # Combine expected free energy and prior
            policies = self._softmax(-expected_free_energy + np.log(policy_prior))
            
            # Update state
            self.state.policies = policies
            
            return np.argmax(policies)
        except Exception as e:
            logger.error(f"Error selecting action: {str(e)}")
            raise
    
    def _compute_expected_free_energy(self) -> np.ndarray:
        """Compute expected free energy for each policy."""
        try:
            n_policies = len(self.state.beliefs)
            expected_free_energy = np.zeros(n_policies)
            
            for i in range(n_policies):
                # Compute predicted next state
                predicted_state = np.dot(self.B[:, :, i], self.state.beliefs)
                
                # Compute predicted observation
                predicted_obs = np.dot(self.A, predicted_state)
                
                # Compute expected free energy components
                ambiguity = -np.sum(predicted_obs * np.log(predicted_obs + 1e-8))
                risk = np.sum(predicted_state * self.C)
                
                expected_free_energy[i] = ambiguity + risk
            
            return expected_free_energy
        except Exception as e:
            logger.error(f"Error computing expected free energy: {str(e)}")
            raise
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values."""
        try:
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)
        except Exception as e:
            logger.error(f"Error computing softmax: {str(e)}")
            raise
    
    def update_precision(self, beta: float = 0.9) -> float:
        """Update precision based on prediction errors."""
        try:
            self.state.precision = (
                beta * self.state.precision +
                (1 - beta) * (1.0 / (self.state.prediction_error + 1e-8))
            )
            return self.state.precision
        except Exception as e:
            logger.error(f"Error updating precision: {str(e)}")
            raise 