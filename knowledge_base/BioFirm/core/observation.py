"""
Observation model implementation for BioFirm framework.
Handles the mapping between true states and observations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import multivariate_normal

from .state_spaces import StateSpace, BioregionalState

@dataclass
class ObservationModel:
    """Generalized observation model for active inference."""
    state_space: StateSpace
    observation_space: StateSpace
    likelihood_matrix: np.ndarray  # A matrix
    noise_model: str = "gaussian"
    precision: float = 1.0
    
    def __post_init__(self):
        """Validate observation model configuration."""
        self._validate_spaces()
        self._validate_likelihood()
        self._setup_noise_model()
        
    def _validate_spaces(self):
        """Ensure state and observation spaces are compatible."""
        if not isinstance(self.state_space, StateSpace):
            raise TypeError("state_space must be a StateSpace instance")
        if not isinstance(self.observation_space, StateSpace):
            raise TypeError("observation_space must be a StateSpace instance")
            
    def _validate_likelihood(self):
        """Validate likelihood matrix dimensions."""
        expected_shape = (
            np.prod(self.observation_space.dimensions),
            np.prod(self.state_space.dimensions)
        )
        if self.likelihood_matrix.shape != expected_shape:
            raise ValueError(
                f"Likelihood matrix shape {self.likelihood_matrix.shape} "
                f"does not match expected shape {expected_shape}"
            )
            
    def _setup_noise_model(self):
        """Configure noise model based on specification."""
        if self.noise_model not in ["gaussian", "poisson", "categorical"]:
            raise ValueError(f"Unsupported noise model: {self.noise_model}")
            
        self.noise_dist = {
            "gaussian": self._gaussian_noise,
            "poisson": self._poisson_noise,
            "categorical": self._categorical_noise
        }[self.noise_model]
        
    def observe(self, state: BioregionalState) -> np.ndarray:
        """Generate observation from true state."""
        # Convert state to vector form
        state_vector = state.to_vector()
        
        # Generate noiseless prediction
        prediction = np.dot(self.likelihood_matrix, state_vector)
        
        # Add noise according to model
        observation = self.noise_dist(prediction)
        
        return observation
        
    def _gaussian_noise(self, prediction: np.ndarray) -> np.ndarray:
        """Apply Gaussian observation noise."""
        noise_cov = np.eye(len(prediction)) / self.precision
        return multivariate_normal.rvs(
            mean=prediction,
            cov=noise_cov
        )
        
    def _poisson_noise(self, prediction: np.ndarray) -> np.ndarray:
        """Apply Poisson observation noise."""
        return np.random.poisson(prediction * self.precision)
        
    def _categorical_noise(self, prediction: np.ndarray) -> np.ndarray:
        """Apply categorical observation noise."""
        # Normalize prediction to probabilities
        probs = prediction / np.sum(prediction)
        
        # Sample categorical distribution
        return np.random.multinomial(1, probs)

class HierarchicalObservation:
    """Hierarchical observation model for multi-scale inference."""
    
    def __init__(self, 
                 models: Dict[str, ObservationModel],
                 scale_couplings: Dict[Tuple[str, str], float]):
        """
        Initialize hierarchical observation model.
        
        Args:
            models: Dictionary mapping scales to ObservationModels
            scale_couplings: Dictionary mapping scale pairs to coupling strengths
        """
        self.models = models
        self.scale_couplings = scale_couplings
        self._validate_configuration()
        
    def _validate_configuration(self):
        """Validate hierarchical model configuration."""
        # Validate models
        if not all(isinstance(m, ObservationModel) for m in self.models.values()):
            raise TypeError("All models must be ObservationModel instances")
            
        # Validate couplings
        scales = set(self.models.keys())
        for (scale1, scale2) in self.scale_couplings.keys():
            if scale1 not in scales or scale2 not in scales:
                raise ValueError(f"Invalid scales in coupling: {scale1}, {scale2}")
                
    def observe_hierarchy(self, 
                         states: Dict[str, BioregionalState]
                         ) -> Dict[str, np.ndarray]:
        """Generate coupled observations across scales."""
        observations = {}
        
        # First pass: generate independent observations
        for scale, state in states.items():
            observations[scale] = self.models[scale].observe(state)
            
        # Second pass: apply cross-scale coupling
        coupled_observations = observations.copy()
        for (scale1, scale2), coupling in self.scale_couplings.items():
            # Compute coupled observation as weighted average
            coupled_observations[scale1] = (
                (1 - coupling) * observations[scale1] +
                coupling * observations[scale2]
            )
            
        return coupled_observations

class ObservationAggregator:
    """Aggregates observations across time and space."""
    
    def __init__(self, 
                 temporal_window: int,
                 spatial_scales: List[str],
                 aggregation_method: str = "mean"):
        """
        Initialize observation aggregator.
        
        Args:
            temporal_window: Number of time steps to aggregate
            spatial_scales: List of spatial scales to consider
            aggregation_method: How to aggregate observations
        """
        self.temporal_window = temporal_window
        self.spatial_scales = spatial_scales
        self.aggregation_method = aggregation_method
        self.observation_buffer = {
            scale: [] for scale in spatial_scales
        }
        
    def add_observation(self, 
                       scale: str,
                       observation: np.ndarray,
                       timestamp: Optional[float] = None):
        """Add new observation to buffer."""
        if scale not in self.spatial_scales:
            raise ValueError(f"Invalid scale: {scale}")
            
        self.observation_buffer[scale].append({
            'observation': observation,
            'timestamp': timestamp or 0.0
        })
        
        # Maintain buffer size
        if len(self.observation_buffer[scale]) > self.temporal_window:
            self.observation_buffer[scale].pop(0)
            
    def get_aggregate(self, scale: str) -> np.ndarray:
        """Get aggregated observation for scale."""
        if scale not in self.spatial_scales:
            raise ValueError(f"Invalid scale: {scale}")
            
        observations = [
            item['observation'] 
            for item in self.observation_buffer[scale]
        ]
        
        if not observations:
            raise ValueError(f"No observations for scale {scale}")
            
        if self.aggregation_method == "mean":
            return np.mean(observations, axis=0)
        elif self.aggregation_method == "median":
            return np.median(observations, axis=0)
        elif self.aggregation_method == "max":
            return np.max(observations, axis=0)
        else:
            raise ValueError(f"Invalid aggregation method: {self.aggregation_method}") 