"""
Generic POMDP Implementation.

This module implements a generic Partially Observable Markov Decision Process (POMDP)
using Active Inference principles. The implementation is designed to be flexible and
handle arbitrary discrete observation and state spaces.

Key Components:
    - A matrix: Observation model P(o|s) mapping states to observations
    - B matrix: Transition model P(s'|s,a) defining state dynamics under actions
    - C matrix: Preference matrix over observations across time
    - D matrix: Prior beliefs over initial states
    - E matrix: Prior preferences over policies
"""

import numpy as np
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

@dataclass
class ModelState:
    """Class to hold the current state of the POMDP."""
    beliefs: np.ndarray
    time_step: int = 0
    current_state: Optional[int] = None
    history: Dict[str, List] = field(default_factory=lambda: {
        'observations': [],
        'actions': [],
        'beliefs': [],
        'free_energy': [],
        'efe_components': []  # Store EFE components for visualization
    })

    def __post_init__(self):
        """Validate state after initialization."""
        if len(self.beliefs.shape) != 1:
            raise ValueError("Beliefs must be a 1D array")
        if not np.allclose(self.beliefs.sum(), 1.0):
            raise ValueError("Beliefs must sum to 1")
        if not np.all(self.beliefs >= 0):
            raise ValueError("Beliefs must be non-negative")
            
    def update(self, observation: int, action: Optional[int], free_energy: float,
              efe_components: Optional[Dict] = None) -> None:
        """Update state and history with new information."""
        if action is not None:
            self.history['actions'].append(int(action))
        self.history['observations'].append(int(observation))
        self.history['beliefs'].append(self.beliefs.copy())
        self.history['free_energy'].append(float(free_energy))
        if efe_components is not None:
            self.history['efe_components'].append(efe_components)
        self.time_step += 1

class GenericPOMDP:
    """Generic POMDP implementation using Active Inference.
    
    This class implements a Partially Observable Markov Decision Process using
    Active Inference principles. It maintains beliefs over hidden states and
    updates them based on observations and actions.
    
    Key Features:
        - Belief updating using variational inference
        - Action selection using Expected Free Energy minimization
        - Temporal preference learning
        - Numerical stability handling
        - State/history saving and loading
        
    Matrix Dimensions:
        - A: (num_observations, num_states) - Column stochastic
        - B: (num_states, num_states, num_actions) - Column stochastic per action
        - C: (num_observations, planning_horizon) - Preference values
        - D: (num_states,) - Initial belief distribution
        - E: (num_actions,) - Policy prior distribution
    
    Attributes:
        num_observations (int): Number of possible observations
        num_states (int): Number of hidden states
        num_actions (int): Number of possible actions
        planning_horizon (int): Number of timesteps to plan ahead
        max_policies (int): Maximum number of policies to evaluate
        temperature (float): Temperature parameter for action selection
        learning_rate (float): Learning rate for belief updates
        stability_threshold (float): Small constant for numerical stability
        state (ModelState): Current state of the POMDP
    """
    
    def __init__(
        self,
        num_observations: int,
        num_states: int,
        num_actions: int,
        planning_horizon: int = 4,
        max_policies: int = 1000,
        temperature: float = 1.0,
        learning_rate: float = 0.1,
        stability_threshold: float = 1e-16
    ):
        """Initialize the POMDP.
        
        Args:
            num_observations: Number of possible observations
            num_states: Number of hidden states
            num_actions: Number of possible actions
            planning_horizon: Number of timesteps to plan ahead (default: 4)
            max_policies: Maximum number of policies to evaluate
            temperature: Temperature parameter for action selection
                        Higher values lead to more exploration
            learning_rate: Learning rate for belief updates
                         Higher values lead to faster but potentially less stable updates
            stability_threshold: Small constant for numerical stability
                               Used to prevent division by zero and log(0)
        
        Raises:
            ValueError: If any dimension is not positive
        """
        # Validate inputs
        if any(x <= 0 for x in [num_observations, num_states, num_actions]):
            raise ValueError("Dimensions must be positive integers")
            
        self.num_observations = num_observations
        self.num_states = num_states
        self.num_actions = num_actions
        self.planning_horizon = planning_horizon
        self.max_policies = max_policies
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.stability_threshold = stability_threshold
        
        # Initialize matrices
        self._initialize_matrices()
        
        # Initialize state
        self.state = ModelState(
            current_state=0,
            beliefs=self.D.copy(),
            time_step=0
        )
        self.state.history['beliefs'].append(self.state.beliefs.copy())
        
    def _initialize_matrices(self):
        """Initialize the model matrices with reasonable defaults."""
        # A matrix: Observation model P(o|s)
        self.A = np.random.rand(self.num_observations, self.num_states)
        self.A = self._normalize(self.A)  # Column stochastic
        
        # B matrix: Transition model P(s'|s,a)
        self.B = np.zeros((self.num_states, self.num_states, self.num_actions))
        for a in range(self.num_actions):
            B_a = np.eye(self.num_states) * 0.8  # Identity-based with high self-transition
            B_a += np.random.rand(self.num_states, self.num_states) * 0.2  # Add random transitions
            self.B[:,:,a] = self._normalize(B_a)  # Column stochastic
        
        # C matrix: Preferences over observations (fixed through time)
        self.C = np.zeros((self.num_observations, self.planning_horizon))  # Initialize with zeros
        # Set strong preference for observation 0
        base_preferences = np.zeros(self.num_observations)
        base_preferences[0] = 2.0  # Strong positive preference for observation 0
        base_preferences[1:] = -0.5  # Slight negative preference for other observations
        for t in range(self.planning_horizon):
            self.C[:, t] = base_preferences
        
        # D matrix: Prior beliefs over states
        self.D = np.ones(self.num_states) / self.num_states  # Uniform prior
        
        # E matrix: Prior over policies
        self.E = np.ones(self.num_actions) / self.num_actions  # Uniform prior
        
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize array to sum to 1 along first axis with numerical stability.
        
        Args:
            x: Array to normalize
            
        Returns:
            Normalized array
        """
        # Handle negative values
        x = np.maximum(x, self.stability_threshold)
        
        if x.ndim == 1:
            # For 1D arrays (e.g. beliefs)
            normalizer = x.sum() + self.stability_threshold
            if normalizer > self.stability_threshold:
                return x / normalizer
            return np.ones_like(x) / len(x)
        else:
            # For 2D arrays (e.g. matrices)
            normalizer = x.sum(axis=0, keepdims=True) + self.stability_threshold
            zero_cols = (normalizer < self.stability_threshold).flatten()
            if np.any(zero_cols):
                x[:, zero_cols] = 1.0 / x.shape[0]
            return x / normalizer
        
    def _compute_free_energy(self, beliefs: np.ndarray) -> float:
        """Compute variational free energy with numerical stability."""
        # Expected log-likelihood
        pred_obs = self.A @ beliefs  # Shape: (num_observations,)
        pred_obs = np.maximum(pred_obs, self.stability_threshold)
        
        # Compute log likelihood using the predicted observations
        log_likelihood = np.log(pred_obs + self.stability_threshold)  # Shape: (num_observations,)
        expected_ll = np.sum(pred_obs * log_likelihood)  # Scalar
        
        # KL divergence from prior
        beliefs = np.maximum(beliefs, self.stability_threshold)
        prior = np.maximum(self.D, self.stability_threshold)
        kl_div = np.sum(beliefs * (np.log(beliefs + self.stability_threshold) - np.log(prior + self.stability_threshold)))
        
        return -expected_ll + kl_div
        
    def _update_beliefs(self, observation: int, action: int) -> np.ndarray:
        """Update beliefs using variational inference with momentum and adaptive learning rate.
        
        This method implements belief updating using a gradient-based approach with
        several numerical stability improvements:
        1. Momentum to help escape local minima
        2. Adaptive learning rate based on belief certainty
        3. Numerical stability thresholds
        4. Convergence checking
        
        Args:
            observation: The observed state
            action: The action taken
            
        Returns:
            Updated beliefs over states
            
        Implementation Details:
            - Uses Bayes rule with iterative optimization
            - Includes momentum term for better convergence
            - Adapts learning rate based on belief certainty
            - Ensures numerical stability throughout
            - Monitors convergence with early stopping
        """
        # Get likelihood and transition matrices
        likelihood = self.A[observation, :]  # Observation likelihood
        transition = self.B[:, :, action]  # State transition matrix
        
        # Compute predicted beliefs (prior)
        predicted_beliefs = transition @ self.state.beliefs
        predicted_beliefs = self._normalize(predicted_beliefs)
        
        # Initialize variables for iterative update
        current_beliefs = predicted_beliefs.copy()
        prev_delta = np.zeros_like(current_beliefs)
        momentum = 0.9
        max_iters = 50
        min_delta = 1e-6
        base_lr = 0.1
        
        for _ in range(max_iters):
            # Store old beliefs for convergence check
            old_beliefs = current_beliefs.copy()
            
            # Compute posterior using Bayes rule
            posterior = likelihood * current_beliefs
            posterior = self._normalize(posterior)
            
            # Compute gradient
            gradient = posterior - current_beliefs
            
            # Adaptive learning rate based on belief certainty
            certainty = 1.0 - self._compute_entropy(current_beliefs) / np.log(len(current_beliefs))
            lr = base_lr * (1.0 - 0.9 * certainty)
            
            # Update with momentum
            delta = lr * gradient + momentum * prev_delta
            current_beliefs += delta
            prev_delta = delta
            
            # Normalize
            current_beliefs = self._normalize(current_beliefs)
            
            # Check convergence
            if np.max(np.abs(current_beliefs - old_beliefs)) < min_delta:
                break
        
        return current_beliefs
        
    def _compute_expected_free_energy(
        self,
        beliefs: np.ndarray,
        action: Optional[int] = None,
        timestep: Optional[int] = None
    ) -> Union[float, Dict[str, np.ndarray]]:
        """Compute expected free energy components.

        Args:
            beliefs: Current belief state.
            action: Optional specific action to compute EFE for.
            timestep: Optional timestep for temporal discounting.

        Returns:
            If action is provided: float representing total EFE for that action
            If action is None: Dictionary containing EFE components for all actions
        """
        # If computing for a single action, return scalar total EFE
        if action is not None:
            # Get next state distribution
            next_beliefs = self.B[:, :, action] @ beliefs

            # Get predicted observations
            predicted_obs = self.A @ next_beliefs

            # Calculate ambiguity (negative entropy of beliefs)
            # Use stability threshold for consistency
            ambiguity = -np.sum(next_beliefs * np.log(next_beliefs + self.stability_threshold))

            # Calculate risk (KL divergence from predicted to preferred outcomes)
            risk = 0.00001 * np.sum(predicted_obs * np.log((predicted_obs + self.stability_threshold) / (predicted_obs + self.stability_threshold)))

            # Calculate expected preferences with proper timestep wrapping
            t = 0 if timestep is None else min(timestep, self.C.shape[1] - 1)
            expected_preferences = 2.0 * np.sum(predicted_obs * self.C[:, t])  # Increased preference weight

            # Return total EFE as scalar
            return ambiguity + risk - expected_preferences

        # Otherwise compute components for all actions
        num_actions = self.num_actions
        ambiguity = np.zeros(num_actions)
        risk = np.zeros(num_actions)
        expected_preferences = np.zeros(num_actions)
        total_efe = np.zeros(num_actions)

        for a in range(num_actions):
            # Get next state distribution
            next_beliefs = self.B[:, :, a] @ beliefs

            # Get predicted observations
            predicted_obs = self.A @ next_beliefs

            # Calculate ambiguity (negative entropy of beliefs)
            ambiguity[a] = -np.sum(next_beliefs * np.log(next_beliefs + self.stability_threshold))

            # Calculate risk (KL divergence from predicted to preferred outcomes)
            risk[a] = 0.00001 * np.sum(predicted_obs * np.log((predicted_obs + self.stability_threshold) / (predicted_obs + self.stability_threshold)))

            # Calculate expected preferences with proper timestep wrapping
            t = 0 if timestep is None else min(timestep, self.C.shape[1] - 1)
            expected_preferences[a] = 2.0 * np.sum(predicted_obs * self.C[:, t])  # Increased preference weight

            # Calculate total EFE
            total_efe[a] = ambiguity[a] + risk[a] - expected_preferences[a]

        return {
            'ambiguity': ambiguity,
            'risk': risk,
            'expected_preferences': expected_preferences,
            'total_efe': total_efe
        }

    def _select_action(self, action_values):
        """Select action using softmax with very low temperature."""
        # Use extremely low temperature to make selection more deterministic
        action_probs = self._softmax(-action_values / (0.000001 + self.stability_threshold))
        return np.random.choice(len(action_probs), p=action_probs)
        
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities with numerical stability.
        
        Args:
            x: Input array
            
        Returns:
            Softmax probabilities
        """
        # Clip values to prevent overflow/underflow
        x = np.clip(x, -500, 500)
        
        # Shift for numerical stability
        x_shifted = x - np.max(x)
        
        # Compute exponentials
        exp_x = np.exp(x_shifted)
        
        # Normalize with stability threshold
        return exp_x / (exp_x.sum() + self.stability_threshold)
        
    def _generate_observation(self) -> int:
        """Generate observation based on current beliefs."""
        # Expected observation distribution
        obs_probs = self.A @ self.state.beliefs
        obs_probs = self._normalize(obs_probs)  # Ensure valid distribution
        
        # Sample observation
        return np.random.choice(self.num_observations, p=obs_probs)
        
    def step(self, action: Optional[int] = None) -> Tuple[int, float]:
        """Take a step in the environment.
        
        Args:
            action: Optional action to take. If None, select action based on policy.
            
        Returns:
            Tuple of (observation, free energy).
        """
        # Select action if not provided
        if action is None:
            # Get EFE components and policy probabilities
            components = self.get_efe_components(self.state.beliefs, self.state.time_step)
            action_posterior = components['action_posterior']
            
            # Sample action from posterior
            action = np.random.choice(self.num_actions, p=action_posterior)
        
        # Get next state distribution
        next_beliefs = self.B[:, :, action] @ self.state.beliefs
        
        # Get observation probabilities
        obs_probs = self.A @ next_beliefs
        
        # Sample observation
        observation = np.random.choice(self.num_observations, p=obs_probs)
        
        # Update beliefs
        updated_beliefs = self._update_beliefs(observation, action)
        
        # Update state
        self.state.beliefs = updated_beliefs
        self.state.time_step += 1
        
        # Calculate free energy for monitoring
        efe_components = self._compute_expected_free_energy(updated_beliefs, action)
        
        # Handle both scalar and dictionary return types
        if isinstance(efe_components, dict):
            free_energy = np.min(efe_components['total_efe'])
        else:
            free_energy = efe_components

        # Update history
        if 'observations' not in self.state.history:
            self.state.history['observations'] = []
        if 'actions' not in self.state.history:
            self.state.history['actions'] = []
        if 'free_energy' not in self.state.history:
            self.state.history['free_energy'] = []
        if 'beliefs' not in self.state.history:
            self.state.history['beliefs'] = []

        self.state.history['observations'].append(observation)
        self.state.history['actions'].append(action)
        self.state.history['free_energy'].append(free_energy)
        self.state.history['beliefs'].append(updated_beliefs.copy())

        return observation, free_energy
        
    def save_state(self, path: Union[str, Path]) -> None:
        """Save current state to file.
        
        Args:
            path: Path to save state to
        """
        path = Path(path)
        state_dict = {
            'beliefs': self.state.beliefs.tolist(),
            'time_step': int(self.state.time_step),
            'current_state': None if self.state.current_state is None else int(self.state.current_state),
            'history': {
                'observations': [int(x) for x in self.state.history['observations']],
                'actions': [None if x is None else int(x) for x in self.state.history['actions']],
                'beliefs': [b.tolist() for b in self.state.history['beliefs']],
                'free_energy': [float(x) for x in self.state.history['free_energy']],
                'efe_components': [
                    {k: v.tolist() if isinstance(v, np.ndarray) else 
                        ([x.tolist() if isinstance(x, np.ndarray) else x for x in v] if isinstance(v, list) else v)
                     for k, v in comp.items()}
                    for comp in self.state.history['efe_components']
                ]
            }
        }
        
        with open(path, 'w') as f:
            json.dump(state_dict, f, indent=2, cls=NumpyEncoder)
            
    def load_state(self, path: Union[str, Path]) -> None:
        """Load state from file.
        
        Args:
            path: Path to load state from
        """
        path = Path(path)
        with open(path, 'r') as f:
            state_dict = json.load(f)
        
        # Create new state with loaded beliefs
        self.state = ModelState(
            beliefs=np.array(state_dict['beliefs']),
            time_step=int(state_dict['time_step']),
            current_state=None if state_dict['current_state'] is None else int(state_dict['current_state'])
        )
        
        # Load history with proper type conversion
        self.state.history = {
            'observations': [int(x) for x in state_dict['history']['observations']],
            'actions': [None if x is None else int(x) for x in state_dict['history']['actions']],
            'beliefs': [np.array(b) for b in state_dict['history']['beliefs']],
            'free_energy': [float(x) for x in state_dict['history']['free_energy']],
            'efe_components': [
                {k: (np.array(v) if isinstance(v, list) and not isinstance(v[0], list)
                     else [np.array(x) if isinstance(x, list) else x for x in v] if isinstance(v, list)
                     else v)
                 for k, v in comp.items()}
                for comp in state_dict['history']['efe_components']
            ]
        }
        
    def reset(self) -> None:
        """Reset the model to initial state."""
        self.state = ModelState(
            current_state=0,
            beliefs=self.D.copy(),
            time_step=0
        )
        self.state.history['beliefs'].append(self.state.beliefs.copy())

    def _compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence between two distributions.
        
        Args:
            p: First distribution
            q: Second distribution
            
        Returns:
            KL divergence value
        """
        # Add stability threshold to avoid log(0)
        p = p + self.stability_threshold
        q = q + self.stability_threshold
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        # Compute KL divergence
        return np.sum(p * (np.log(p) - np.log(q)))

    def get_efe_components(
        self,
        beliefs: Optional[np.ndarray] = None,
        timestep: Optional[int] = None
    ) -> Dict[str, Union[np.ndarray, List[List[int]]]]:
        """Get expected free energy components for all policies.

        This method computes the Expected Free Energy (EFE) components for all possible
        policies up to the planning horizon. The EFE is composed of:
        1. Ambiguity: Entropy of beliefs (epistemic value)
        2. Risk: KL divergence between predicted and preferred observations
        3. Expected Preferences: Expected value of future observations
        
        The total EFE is used to compute action probabilities through softmax.

        Args:
            beliefs: Optional beliefs to use instead of current state
            timestep: Optional timestep for temporal discounting

        Returns:
            Dictionary containing:
            - policies: List of action sequences
            - ambiguity: Epistemic value term
            - risk: Risk term from KL divergence
            - expected_preferences: Expected preference satisfaction
            - total_efe: Sum of components
            - action_posterior: Action selection probabilities
            
        Implementation Details:
            - Generates all possible policies up to planning horizon
            - Computes EFE components for each policy
            - Uses temporal discounting for future preferences
            - Ensures numerical stability in all computations
            - Normalizes distributions appropriately
        """
        if beliefs is None:
            beliefs = self.state.beliefs
        if timestep is None:
            timestep = self.state.time_step

        # Generate policies as lists for proper temporal discounting
        policies = [[a] for a in range(self.num_actions)]
        for _ in range(1, self.planning_horizon):
            new_policies = []
            for policy in policies:
                for a in range(self.num_actions):
                    new_policies.append(policy + [a])
            policies = new_policies

        num_policies = len(policies)
        ambiguity = np.zeros((num_policies, self.planning_horizon))
        risk = np.zeros((num_policies, self.planning_horizon))
        expected_preferences = np.zeros((num_policies, self.planning_horizon))
        total_efe = np.zeros(num_policies)

        # Compute components for each policy
        for p, policy in enumerate(policies):
            curr_beliefs = beliefs.copy()
            
            for t, action in enumerate(policy):
                # Get next state distribution
                next_beliefs = self.B[:, :, action] @ curr_beliefs
                
                # Get predicted observations
                pred_obs = self.A @ next_beliefs
                
                # Compute ambiguity (entropy of beliefs)
                ambiguity[p, t] = -np.sum(next_beliefs * np.log(next_beliefs + self.stability_threshold))
                
                # Compute risk (KL between predicted and preferred outcomes)
                preferred_obs = self._softmax(self.C[:, t] if t < self.C.shape[1] else self.C[:, -1])
                risk[p, t] = self._compute_kl_divergence(pred_obs, preferred_obs)
                
                # Store expected preferences
                expected_preferences[p, t] = np.sum(pred_obs * self.C[:, t if t < self.C.shape[1] else -1])
                
                # Update beliefs for next timestep
                curr_beliefs = next_beliefs

            # Compute total EFE for policy with temporal discounting
            total_efe[p] = np.sum(ambiguity[p, :] + risk[p, :] - expected_preferences[p, :])

        # Compute action posterior using softmax
        action_posterior = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            # Sum EFE for all policies starting with action a
            policy_indices = [i for i, p in enumerate(policies) if p[0] == a]
            action_posterior[a] = np.sum(self._softmax(-total_efe)[policy_indices])
        
        # Normalize action posterior
        action_posterior = self._normalize(action_posterior)

        return {
            'policies': policies,
            'ambiguity': ambiguity,
            'risk': risk,
            'expected_preferences': expected_preferences,
            'total_efe': total_efe,
            'action_posterior': action_posterior
        }

    def _compute_entropy(self, dist: np.ndarray) -> float:
        """Compute entropy of a distribution with numerical stability.
        
        Args:
            dist: Probability distribution
            
        Returns:
            Entropy value
        """
        # Ensure valid probabilities
        dist = np.maximum(dist, self.stability_threshold)
        dist = dist / (dist.sum() + self.stability_threshold)
        
        # Compute entropy with clipping to prevent log(0)
        return -np.sum(dist * np.log(dist + self.stability_threshold))

    def test_action_selection(self):
        """Test that action selection uses policy evaluation properly."""
        # Run action selection multiple times
        n_samples = 100
        selected_actions = []
        action_probs_list = []

        for _ in range(n_samples):
            # Get EFE components to compute action values
            efe_components = self.get_efe_components()
            action_values = -efe_components['total_efe']  # Negative because we want to minimize EFE
            
            # Get action probabilities
            action_probs = self._softmax(-action_values / (0.00001 + self.stability_threshold))
            action = np.random.choice(len(action_probs), p=action_probs)
            
            selected_actions.append(action)
            action_probs_list.append(action_probs)

        # Check that actions are selected with appropriate probabilities
        action_counts = np.bincount(selected_actions, minlength=self.num_actions)
        action_frequencies = action_counts / n_samples

        # Average action probabilities across samples
        avg_action_probs = np.mean(action_probs_list, axis=0)

        # Check that empirical frequencies roughly match predicted probabilities
        assert np.allclose(action_frequencies, avg_action_probs, atol=0.1)

        return action, action_probs 