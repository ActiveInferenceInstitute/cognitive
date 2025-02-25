"""
Active Inference method dispatcher and high-level abstractions.
Provides a clean interface for dispatching active inference operations
to appropriate low-level implementations.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from .base import ActiveInferenceModel, ModelState
from ..matrices.matrix_ops import MatrixOps, MatrixInitializer

class InferenceMethod(Enum):
    """Supported inference methods."""
    VARIATIONAL = "variational"
    SAMPLING = "sampling"
    MEAN_FIELD = "mean_field"
    
class PolicyType(Enum):
    """Supported policy types."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    HIERARCHICAL = "hierarchical"

@dataclass
class InferenceConfig:
    """Configuration for inference method dispatch."""
    method: InferenceMethod
    policy_type: PolicyType
    temporal_horizon: int
    learning_rate: float
    precision_init: float
    use_gpu: bool = False
    num_samples: int = 1000  # For sampling-based methods
    temperature: float = 1.0  # For policy selection
    custom_params: Optional[Dict[str, Any]] = None

class ActiveInferenceDispatcher:
    """
    Dispatcher for Active Inference operations.
    Provides high-level interface and handles routing to specific implementations.
    """
    
    def __init__(self, config: InferenceConfig):
        """Initialize dispatcher with configuration."""
        self.config = config
        self._setup_implementations()
        self._initialize_matrices()
        self._rng = np.random.default_rng()  # For sampling methods
        
    def _setup_implementations(self):
        """Set up mapping of operations to implementations."""
        self._implementations = {
            InferenceMethod.VARIATIONAL: {
                'belief_update': self._variational_belief_update,
                'policy_inference': self._variational_policy_inference
            },
            InferenceMethod.SAMPLING: {
                'belief_update': self._sampling_belief_update,
                'policy_inference': self._sampling_policy_inference
            },
            InferenceMethod.MEAN_FIELD: {
                'belief_update': self._mean_field_belief_update,
                'policy_inference': self._mean_field_policy_inference
            }
        }
        
    def _initialize_matrices(self):
        """Initialize required matrices based on configuration."""
        self.matrix_ops = MatrixOps()
        self.matrix_init = MatrixInitializer()
        
    def dispatch_belief_update(self, 
                             observation: np.ndarray,
                             current_state: ModelState,
                             **kwargs) -> np.ndarray:
        """
        Dispatch belief update to appropriate implementation.
        
        Args:
            observation: Current observation
            current_state: Current model state
            **kwargs: Additional parameters for specific implementations
            
        Returns:
            Updated beliefs
        """
        update_fn = self._implementations[self.config.method]['belief_update']
        return update_fn(observation, current_state, **kwargs)
        
    def dispatch_policy_inference(self,
                                state: ModelState,
                                goal_prior: Optional[np.ndarray] = None,
                                **kwargs) -> np.ndarray:
        """
        Dispatch policy inference to appropriate implementation.
        
        Args:
            state: Current model state
            goal_prior: Optional prior over goal states
            **kwargs: Additional parameters for specific implementations
            
        Returns:
            Inferred policy distributions
        """
        inference_fn = self._implementations[self.config.method]['policy_inference']
        return inference_fn(state, goal_prior, **kwargs)
    
    def _variational_belief_update(self,
                                 observation: np.ndarray,
                                 state: ModelState,
                                 **kwargs) -> np.ndarray:
        """Variational implementation of belief updates."""
        # Implementation details for variational belief updates
        prediction = np.dot(state.beliefs, kwargs.get('generative_matrix', np.eye(len(state.beliefs))))
        prediction_error = observation - prediction
        belief_update = state.precision * prediction_error
        return state.beliefs + belief_update
        
    def _sampling_belief_update(self,
                              observation: np.ndarray,
                              state: ModelState,
                              **kwargs) -> np.ndarray:
        """Sampling-based implementation of belief updates using particle filtering."""
        num_samples = self.config.num_samples
        generative_matrix = kwargs.get('generative_matrix', np.eye(len(state.beliefs)))
        
        # Initialize particles
        particles = self._rng.dirichlet(
            state.beliefs * num_samples,
            size=num_samples
        )
        
        # Compute weights based on likelihood
        likelihoods = np.array([
            self._compute_likelihood(observation, p, generative_matrix)
            for p in particles
        ])
        weights = likelihoods / np.sum(likelihoods)
        
        # Resample particles
        resampled_indices = self._rng.choice(
            num_samples,
            size=num_samples,
            p=weights
        )
        particles = particles[resampled_indices]
        
        # Return mean belief state
        return np.mean(particles, axis=0)
        
    def _compute_likelihood(self,
                          observation: np.ndarray,
                          particle: np.ndarray,
                          generative_matrix: np.ndarray) -> float:
        """Compute likelihood of observation given particle state."""
        prediction = np.dot(particle, generative_matrix)
        return np.exp(-0.5 * np.sum(np.square(observation - prediction)))
        
    def _mean_field_belief_update(self,
                                observation: np.ndarray,
                                state: ModelState,
                                **kwargs) -> np.ndarray:
        """Mean-field implementation of belief updates."""
        # Implementation for mean-field updates
        raise NotImplementedError("Mean-field belief updates not yet implemented")
        
    def _variational_policy_inference(self,
                                    state: ModelState,
                                    goal_prior: Optional[np.ndarray] = None,
                                    **kwargs) -> np.ndarray:
        """Variational implementation of policy inference."""
        # Implementation for variational policy inference
        if goal_prior is None:
            goal_prior = np.ones(len(state.policies)) / len(state.policies)
        
        expected_free_energy = self._calculate_expected_free_energy(
            state, goal_prior, **kwargs)
        return self.matrix_ops.softmax(-expected_free_energy)
        
    def _sampling_policy_inference(self,
                                 state: ModelState,
                                 goal_prior: Optional[np.ndarray] = None,
                                 **kwargs) -> np.ndarray:
        """Sampling-based implementation of policy inference using MCMC."""
        if goal_prior is None:
            goal_prior = np.ones(len(state.policies)) / len(state.policies)
            
        num_samples = self.config.num_samples
        current_policies = state.policies.copy()
        accepted_policies = []
        
        # MCMC sampling
        for _ in range(num_samples):
            # Propose new policy distribution
            proposal = self._propose_policy(current_policies)
            
            # Compute acceptance ratio
            current_energy = self._policy_energy(current_policies, state, goal_prior)
            proposal_energy = self._policy_energy(proposal, state, goal_prior)
            
            # Accept/reject
            if np.log(self._rng.random()) < proposal_energy - current_energy:
                current_policies = proposal
            
            accepted_policies.append(current_policies.copy())
            
        # Return mean policy distribution
        return np.mean(accepted_policies, axis=0)
        
    def _propose_policy(self, current: np.ndarray) -> np.ndarray:
        """Generate policy proposal for MCMC."""
        proposal = current + self._rng.normal(0, 0.1, size=current.shape)
        return self.matrix_ops.normalize_rows(np.maximum(proposal, 0))
        
    def _policy_energy(self,
                      policies: np.ndarray,
                      state: ModelState,
                      goal_prior: np.ndarray) -> float:
        """Compute energy (negative log probability) for policy distribution."""
        expected_free_energy = self._calculate_expected_free_energy(
            state, goal_prior)
        return np.sum(policies * expected_free_energy)
        
    def _mean_field_policy_inference(self,
                                   state: ModelState,
                                   goal_prior: Optional[np.ndarray] = None,
                                   **kwargs) -> np.ndarray:
        """Mean-field implementation of policy inference."""
        raise NotImplementedError("Mean-field policy inference not yet implemented")
        
    def _calculate_expected_free_energy(self,
                                      state: ModelState,
                                      goal_prior: np.ndarray,
                                      **kwargs) -> np.ndarray:
        """Calculate expected free energy for policy evaluation."""
        # Enhanced implementation with both pragmatic and epistemic value
        pragmatic_value = self._calculate_pragmatic_value(state, goal_prior)
        epistemic_value = self._calculate_epistemic_value(state)
        
        # Weight between exploration and exploitation
        exploration_weight = kwargs.get('exploration_weight', 0.5)
        return (1 - exploration_weight) * pragmatic_value + exploration_weight * epistemic_value
        
    def _calculate_pragmatic_value(self,
                                 state: ModelState,
                                 goal_prior: np.ndarray) -> np.ndarray:
        """Calculate pragmatic value component of expected free energy."""
        # KL divergence from current state to goal state
        return -np.log(goal_prior + 1e-8)
        
    def _calculate_epistemic_value(self, state: ModelState) -> np.ndarray:
        """Calculate epistemic value component of expected free energy."""
        # Information gain approximation
        uncertainty = -np.sum(state.beliefs * np.log(state.beliefs + 1e-8))
        return -uncertainty * np.ones(len(state.policies))
        
    def update_precision(self, prediction_error: float) -> float:
        """Update precision parameter based on prediction errors."""
        if self.config.method == InferenceMethod.VARIATIONAL:
            # Precision updates for variational method
            self.config.precision_init = (
                0.9 * self.config.precision_init +
                0.1 / (prediction_error + 1e-8)
            )
        elif self.config.method == InferenceMethod.SAMPLING:
            # Adaptive step size for sampling method
            self.config.precision_init = np.clip(
                1.0 / (prediction_error + 1e-8),
                0.1,
                10.0
            )
        return self.config.precision_init

class ActiveInferenceFactory:
    """Factory for creating Active Inference instances with specific configurations."""
    
    @staticmethod
    def create(config: InferenceConfig) -> ActiveInferenceDispatcher:
        """Create an Active Inference dispatcher with specified configuration."""
        return ActiveInferenceDispatcher(config)
    
    @staticmethod
    def create_from_yaml(config_path: Union[str, Path]) -> ActiveInferenceDispatcher:
        """Create an Active Inference dispatcher from YAML configuration."""
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = InferenceConfig(
            method=InferenceMethod(config_dict['method']),
            policy_type=PolicyType(config_dict['policy_type']),
            temporal_horizon=config_dict['temporal_horizon'],
            learning_rate=config_dict['learning_rate'],
            precision_init=config_dict['precision_init'],
            use_gpu=config_dict.get('use_gpu', False),
            num_samples=config_dict.get('num_samples', 1000),
            temperature=config_dict.get('temperature', 1.0),
            custom_params=config_dict.get('custom_params', None)
        )
        return ActiveInferenceFactory.create(config) 