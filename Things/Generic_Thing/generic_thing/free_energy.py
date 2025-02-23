"""
Implementation of Free Energy Principle for Generic Things.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
from scipy.optimize import minimize

@dataclass
class FreeEnergy:
    """
    Implements the Free Energy Principle for Generic Things.
    
    The Free Energy Principle states that all self-organizing systems minimize
    their variational free energy, which bounds surprise (negative log evidence).
    This implementation provides methods for:
    - Computing variational free energy
    - Minimizing free energy through perception and action
    - Maintaining generative models and recognition densities
    """
    
    # Generative model parameters
    model_parameters: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Recognition density parameters
    recognition_parameters: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Historical free energy values
    history: List[float] = field(default_factory=list)
    
    # State variables
    internal_states: Dict[str, np.ndarray] = field(default_factory=dict)
    external_states: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def initialize_states(self, state_shape: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize internal and external states.
        
        Args:
            state_shape: Dictionary specifying the shape of each state variable.
                       If None, uses default shapes.
        """
        if state_shape is None:
            state_shape = {'x': (1,), 'y': (1,)}  # Default shape
            
        for key, shape in state_shape.items():
            self.internal_states[key] = np.zeros(shape)
            self.external_states[key] = np.zeros(shape)
    
    def compute_free_energy(self,
                          internal_state: Dict[str, Any],
                          external_state: Dict[str, Any],
                          observation: Dict[str, Any]) -> float:
        """
        Compute the variational free energy.
        
        Args:
            internal_state: Current internal state
            external_state: Current external state beliefs
            observation: Current sensory observation
            
        Returns:
            Computed free energy value
            
        Raises:
            ValueError: If state dimensions are incompatible
        """
        # Initialize energy terms
        accuracy = 0.0
        complexity = 0.0
        
        # Validate state dimensions
        for key in observation.keys():
            if key not in internal_state:
                raise ValueError(f"Missing internal state variable '{key}'")
            if key not in external_state:
                raise ValueError(f"Missing external state variable '{key}'")
        
        # Compute energy terms for each state variable
        for key in observation.keys():
            obs_val = np.array(observation[key]).flatten()
            ext_val = np.array(external_state[key]).flatten()
            int_val = np.array(internal_state[key]).flatten()
            
            # Ensure all arrays have the same shape
            min_len = min(len(obs_val), len(ext_val), len(int_val))
            obs_val = obs_val[:min_len]
            ext_val = ext_val[:min_len]
            int_val = int_val[:min_len]
            
            # Compute energy terms
            accuracy += np.sum((obs_val - ext_val) ** 2)  # Accuracy term
            complexity += np.sum((ext_val - int_val) ** 2)  # Complexity term
        
        free_energy = accuracy + complexity
        self.history.append(free_energy)
        
        return free_energy
    
    def minimize(self,
                internal_state: Dict[str, Any],
                external_state: Dict[str, Any],
                observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Minimize free energy through perception and action.
        
        Args:
            internal_state: Current internal state
            external_state: Current external state beliefs
            observation: Current sensory observation
            
        Returns:
            Updated external state that minimizes free energy
        """
        # Flatten states for optimization
        x0 = []
        shapes = {}  # Store original shapes
        keys = []    # Store keys in order
        
        for key in observation.keys():
            val = np.array(external_state.get(key, np.zeros_like(observation[key])))
            shapes[key] = val.shape
            x0.extend(val.flatten())
            keys.extend([key] * val.size)
        
        x0 = np.array(x0)
        
        # Define objective function for optimization
        def objective(x):
            # Convert optimization vector back to state dict
            test_state = {}
            idx = 0
            for key, shape in shapes.items():
                size = int(np.prod(shape))  # Ensure size is an integer
                val = x[idx:idx+size].reshape(shape)
                test_state[key] = val
                idx += size
            return self.compute_free_energy(internal_state, test_state, observation)
        
        # Minimize free energy
        result = minimize(objective, x0, method='L-BFGS-B')
        
        # Convert result back to state dict
        optimized_state = {}
        idx = 0
        for key, shape in shapes.items():
            size = int(np.prod(shape))  # Ensure size is an integer
            val = result.x[idx:idx+size].reshape(shape)
            optimized_state[key] = val
            idx += size
        
        return optimized_state
    
    def select_action(self,
                     internal_state: Dict[str, Any],
                     external_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select actions that minimize expected free energy.
        
        Args:
            internal_state: Current internal state
            external_state: Current external state beliefs
            
        Returns:
            Selected actions as a dictionary
        """
        actions = {}
        for key in internal_state.keys():
            # Compute prediction error (external - internal)
            prediction_error = external_state.get(key, 0) - internal_state[key]
            # Action should reduce prediction error by moving external state towards internal state
            # When external state is higher than internal state (positive error), action should be negative
            # When external state is lower than internal state (negative error), action should be positive
            actions[f"action_{key}"] = -prediction_error  # Negative feedback to reduce error
            
        return actions
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the free energy system."""
        return {
            'model_parameters': self.model_parameters,
            'recognition_parameters': self.recognition_parameters,
            'current_free_energy': self.history[-1] if self.history else None,
            'history': self.history
        }
    
    def update_parameters(self,
                        model_params: Optional[Dict[str, np.ndarray]] = None,
                        recog_params: Optional[Dict[str, np.ndarray]] = None) -> None:
        """
        Update the generative model and recognition density parameters.
        
        Args:
            model_params: New generative model parameters
            recog_params: New recognition density parameters
            
        Raises:
            ValueError: If parameters are invalid or have incompatible dimensions
            TypeError: If parameters are not array-convertible
        """
        if model_params is not None:
            # Validate model parameters
            for key, value in model_params.items():
                if value is None:
                    raise ValueError(f"Model parameter '{key}' cannot be None")
                try:
                    value = np.array(value)
                except:
                    raise TypeError(f"Model parameter '{key}' must be convertible to numpy array")
                
                # Check if parameter exists and validate dimensions
                if key in self.model_parameters:
                    if value.shape != self.model_parameters[key].shape:
                        raise ValueError(f"Model parameter '{key}' has incompatible shape: expected {self.model_parameters[key].shape}, got {value.shape}")
            
            self.model_parameters.update(model_params)
            
        if recog_params is not None:
            # Validate recognition parameters
            for key, value in recog_params.items():
                if value is None:
                    raise ValueError(f"Recognition parameter '{key}' cannot be None")
                try:
                    value = np.array(value)
                except:
                    raise TypeError(f"Recognition parameter '{key}' must be convertible to numpy array")
                
                # Check if parameter exists and validate dimensions
                if key in self.recognition_parameters:
                    if value.shape != self.recognition_parameters[key].shape:
                        raise ValueError(f"Recognition parameter '{key}' has incompatible shape: expected {self.recognition_parameters[key].shape}, got {value.shape}")
            
            self.recognition_parameters.update(recog_params)
    
    def compute_expected_free_energy(self,
                                   internal_state: Dict[str, Any],
                                   external_state: Dict[str, Any],
                                   action: Dict[str, Any]) -> float:
        """
        Compute the expected free energy for a given action.
        
        Args:
            internal_state: Current internal state
            external_state: Current external state beliefs
            action: Proposed action to evaluate
            
        Returns:
            Expected free energy value
            
        Raises:
            ValueError: If state dimensions are incompatible
        """
        # Initialize energy terms
        expected_accuracy = 0.0
        expected_complexity = 0.0
        information_gain = 0.0
        
        # For each state variable
        for key in internal_state.keys():
            if key not in external_state:
                raise ValueError(f"Missing external state variable '{key}'")
                
            # Current values
            int_val = np.array(internal_state[key]).flatten()
            ext_val = np.array(external_state[key]).flatten()
            
            # Predicted next state after action
            action_key = f"action_{key}"
            if action_key in action:
                action_val = action[action_key]
                predicted_state = ext_val + action_val
            else:
                predicted_state = ext_val
            
            # Compute energy terms
            expected_accuracy += np.sum((predicted_state - int_val) ** 2)
            expected_complexity += np.sum((predicted_state - ext_val) ** 2)
            
            # Information gain term (if we have recognition parameters)
            if key in self.recognition_parameters:
                uncertainty = np.var(self.recognition_parameters[key])
                # Higher uncertainty should lead to lower EFE to encourage exploration
                information_gain += uncertainty
        
        # Combine terms - weight information gain negatively to encourage exploration of uncertain states
        expected_free_energy = expected_accuracy + expected_complexity - 0.5 * information_gain
        
        return expected_free_energy 