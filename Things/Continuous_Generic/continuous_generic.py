"""
Continuous Generic Active Inference Implementation.

This module implements a continuous-time active inference agent using classical
mathematics and generalized coordinates. The implementation follows the original
formulation using Taylor series approximations for belief updating and free energy
minimization.

Key Components:
    - Generalized coordinates: x = [x, x', x'', ...]
    - Generative model: p(o,x) = p(o|x)p(x)
    - Free Energy: F = -ln p(o,x)
    - Belief updating: dx/dt = D x - ∂F/∂x
    where D is the shift operator in generalized coordinates
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import factorial
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContinuousState:
    """Class to hold the current state of the continuous active inference agent.
    
    The state is represented in generalized coordinates up to order n:
    x = [x, x', x'', ..., x^(n)]
    """
    
    # Generalized coordinates of beliefs (means and precisions)
    belief_means: np.ndarray  # Shape: [n_states, n_orders]
    belief_precisions: np.ndarray  # Shape: [n_states, n_orders]
    
    # Time tracking
    time: float = 0.0
    dt: float = 0.01
    
    # Current true state (if available)
    true_state: Optional[np.ndarray] = None
    
    # History for tracking and visualization
    history: Dict[str, List] = field(default_factory=lambda: {
        'observations': [],
        'actions': [],
        'belief_means': [],
        'belief_precisions': [],
        'free_energy': [],
        'time': [],
        'true_state': []
    })
    
    def __post_init__(self):
        """Validate state after initialization."""
        if self.belief_means.shape != self.belief_precisions.shape:
            raise ValueError("Belief means and precisions must have same shape")
        if not np.all(self.belief_precisions > 0):
            raise ValueError("Belief precisions must be positive")

    def update(self, 
               observation: np.ndarray,
               action: np.ndarray,
               free_energy: float) -> None:
        """Update state and history with new information."""
        self.history['observations'].append(observation.copy())
        self.history['actions'].append(action.copy())
        self.history['belief_means'].append(self.belief_means.copy())
        self.history['belief_precisions'].append(self.belief_precisions.copy())
        self.history['free_energy'].append(float(free_energy))
        self.history['time'].append(float(self.time))
        if self.true_state is not None:
            self.history['true_state'].append(self.true_state.copy())
        self.time += self.dt

class ContinuousActiveInference:
    """Continuous-time active inference using generalized coordinates."""
    
    def __init__(self,
                 n_states: int,
                 n_obs: int,
                 n_orders: int = 3,
                 dt: float = 0.01,
                 alpha: float = 1.0,
                 sigma: float = 1.0):
        """Initialize the continuous active inference agent.
        
        Args:
            n_states: Number of state dimensions
            n_obs: Number of observation dimensions
            n_orders: Number of generalized coordinate orders
            dt: Integration time step
            alpha: Learning rate for belief updating
            sigma: Action selection precision
        """
        self.n_states = n_states
        self.n_obs = n_obs
        self.n_orders = n_orders
        self.dt = dt
        self.alpha = alpha
        self.sigma = sigma
        
        # Initialize state
        self.state = ContinuousState(
            belief_means=np.zeros((n_states, n_orders)),
            belief_precisions=np.ones((n_states, n_orders)),
            dt=dt
        )
        
        # Create shift operator matrix D for generalized coordinates
        self.D = self._create_shift_operator()
        
    def _create_shift_operator(self) -> np.ndarray:
        """Create shift operator matrix for generalized coordinates.
        
        The shift operator D maps between orders of motion:
        D[x, x', x'', ...] = [x', x'', x''', ...]
        
        This implementation includes factorial scaling to properly handle
        the relationship between different orders of motion in the Taylor series.
        """
        D = np.zeros((self.n_orders, self.n_orders))
        for i in range(self.n_orders - 1):
            # Add factorial scaling for proper Taylor series representation
            D[i, i+1] = factorial(i+1) / factorial(i)
        return D
        
    def _sensory_mapping(self, states: np.ndarray) -> np.ndarray:
        """Map states to observations (g(x) in standard notation).
        
        This implements a nonlinear sensory mapping that considers multiple
        orders of motion. The mapping includes cross-terms between states
        to capture more complex relationships.
        """
        # Get lowest order states
        x = states[:, 0]
        
        # Nonlinear observation mapping
        obs = np.zeros(self.n_obs)
        
        # First order terms
        obs = x.copy()
        
        # Add nonlinear terms if we have multiple states
        if self.n_states > 1:
            for i in range(self.n_states):
                for j in range(i+1, self.n_states):
                    # Add cross terms
                    obs[i] += 0.1 * x[i] * x[j]
                    obs[j] += 0.1 * x[i] * x[j]
                    
                # Add quadratic terms
                obs[i] += 0.05 * x[i]**2
                
        return obs
        
    def _flow_mapping(self, states: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Map states and action to state flow (f(x,a) in standard notation).
        
        This implements a nonlinear dynamical system with:
        1. Conservative forces (derived from a potential)
        2. Dissipative forces (friction)
        3. Action-dependent forcing
        4. Cross-coupling between states
        """
        flow = np.zeros_like(states)
        
        # Extract state components
        x = states[:, 0]  # Position
        v = states[:, 1]  # Velocity
        
        # Conservative forces (derived from quadratic potential)
        U = 0.5 * np.sum(x**2)  # Quadratic potential
        F_conservative = -x  # Gradient of quadratic potential
        
        # Dissipative forces (friction)
        F_dissipative = -0.1 * v
        
        # Action-dependent forcing
        F_action = action
        
        # Cross-coupling between states
        F_coupling = np.zeros_like(x)
        if self.n_states > 1:
            for i in range(self.n_states):
                for j in range(self.n_states):
                    if i != j:
                        F_coupling[i] += 0.05 * x[j] * v[j]
        
        # Update flow for each order
        flow[:, 0] = v  # dx/dt = v
        flow[:, 1] = F_conservative + F_dissipative + F_action + F_coupling  # dv/dt = F/m
        
        # Higher order flows (if any)
        for i in range(2, self.n_orders):
            flow[:, i] = np.zeros_like(x)  # Higher orders evolve freely
            
        return flow
        
    def _compute_free_energy(self,
                           obs: np.ndarray,
                           states: np.ndarray,
                           action: Optional[np.ndarray] = None) -> float:
        """Compute variational free energy.
        
        F = -ln p(o,x) = -ln p(o|x) - ln p(x)
        
        This implementation includes:
        1. Accuracy term (prediction error)
        2. Complexity term (KL divergence from prior)
        3. Expected flow term
        4. Entropy term
        """
        # 1. Accuracy term (prediction error)
        pred_obs = self._sensory_mapping(states)
        sensory_pe = 0.5 * np.sum((obs - pred_obs)**2)
        
        # 2. Complexity term (KL divergence from prior)
        # Assume Gaussian prior with zero mean and unit precision
        complexity = 0.5 * np.sum(states[:, 0]**2)
        
        # 3. Expected flow term
        if action is not None:
            pred_flow = self._flow_mapping(states, action)
            # Compare predicted flow with actual generalized motion
            flow_pe = 0.0
            for i in range(self.n_orders-1):
                flow_error = states[:, i+1] - pred_flow[:, i]
                flow_pe += 0.5 * np.sum(flow_error**2)
        else:
            flow_pe = 0.0
            
        # 4. Entropy term (encourage exploration)
        # Use precision-weighted entropy
        entropy = -0.5 * np.sum(np.log(self.state.belief_precisions + 1e-8))
        
        # Combine all terms
        free_energy = sensory_pe + complexity + flow_pe - entropy
        
        return free_energy
        
    def _update_beliefs(self,
                       obs: np.ndarray,
                       action: Optional[np.ndarray] = None,
                       n_steps: int = 10) -> Tuple[np.ndarray, float]:
        """Update beliefs using generalized descent on free energy.
        
        dx/dt = D x - ∂F/∂x
        
        This implementation uses:
        1. Gradient descent with momentum
        2. Adaptive learning rates
        3. Precision weighting of updates
        4. Natural gradient in belief space
        
        Args:
            obs: Observation
            action: Optional action
            n_steps: Number of integration steps
            
        Returns:
            Updated beliefs and free energy
        """
        states = self.state.belief_means.copy()
        velocities = np.zeros_like(states)  # For momentum
        learning_rates = np.ones_like(states)  # Adaptive learning rates
        
        beta1 = 0.9  # Momentum parameter
        beta2 = 0.999  # RMSprop parameter
        epsilon = 1e-8  # Numerical stability
        
        for step in range(n_steps):
            # Compute free energy and gradients
            F = self._compute_free_energy(obs, states, action)
            
            # Compute gradients using natural gradient
            # dF = ∇F = Σ^(-1) * ∂F/∂x where Σ is the precision matrix
            dF = np.zeros_like(states)
            eps = 1e-6
            for i in range(self.n_states):
                for j in range(self.n_orders):
                    states[i,j] += eps
                    Fp = self._compute_free_energy(obs, states, action)
                    states[i,j] -= 2*eps
                    Fm = self._compute_free_energy(obs, states, action)
                    states[i,j] += eps
                    dF[i,j] = (Fp - Fm)/(2*eps)
            
            # Update precisions based on prediction errors
            pred_obs = self._sensory_mapping(states)
            obs_error = obs - pred_obs
            # Update precisions for all orders based on lowest order error
            for i in range(self.n_orders):
                self.state.belief_precisions[:, i] = 1.0 / (obs_error**2 + epsilon)
            
            # Precision weighting of gradients
            dF = dF / (self.state.belief_precisions + epsilon)
            
            # Generalized motion
            dstates = np.dot(states, self.D.T)
            
            # Update velocities (momentum)
            velocities = beta1 * velocities + (1 - beta1) * (dstates - self.alpha * dF)
            
            # Update learning rates (RMSprop)
            learning_rates = beta2 * learning_rates + (1 - beta2) * dF**2
            
            # Compute effective step size
            effective_lr = self.dt / (np.sqrt(learning_rates) + epsilon)
            
            # Update states
            states += effective_lr * velocities
            
        self.state.belief_means = states
        return states, F
        
    def _compute_expected_free_energy(self,
                                    states: np.ndarray,
                                    action: np.ndarray) -> float:
        """Compute expected free energy for action selection.
        
        G = E_q(s')[F(s') + KL[q(s')||p(s')]] - H[q(o'|s')]
        
        This includes:
        1. Expected accuracy
        2. Expected complexity
        3. Expected information gain
        4. Goal-directed behavior
        """
        # Predict next state
        flow = self._flow_mapping(states, action)
        next_states = states + self.dt * flow
        
        # 1. Expected accuracy
        pred_obs = self._sensory_mapping(next_states)
        # Use current observation as proxy for desired observation
        desired_obs = self._sensory_mapping(states)  # Could be replaced with actual goals
        expected_accuracy = 0.5 * np.sum((pred_obs - desired_obs)**2)
        
        # 2. Expected complexity
        expected_complexity = 0.5 * np.sum(next_states[:, 0]**2 + np.log(2*np.pi))
        
        # 3. Expected information gain
        # Approximate using entropy of predicted observations
        pred_var = 1.0 / (self.state.belief_precisions[:, 0] + 1e-8)
        information_gain = 0.5 * np.sum(np.log(2*np.pi*np.e*pred_var))
        
        # 4. Goal-directed term
        # Add specific goals here if available
        goal_term = 0.0
        
        # Combine terms with weights
        w_accuracy = 1.0
        w_complexity = 0.1
        w_info_gain = 0.5
        w_goal = 1.0
        
        G = (w_accuracy * expected_accuracy +
             w_complexity * expected_complexity -
             w_info_gain * information_gain +
             w_goal * goal_term)
             
        return G
        
    def _select_action(self, obs: np.ndarray) -> np.ndarray:
        """Select action by minimizing expected free energy.
        
        This implementation uses:
        1. Path integral control
        2. Multiple action candidates
        3. Momentum-based optimization
        4. Adaptive exploration
        """
        n_candidates = 10  # Number of action candidates
        horizon = 5  # Planning horizon
        
        # Initialize action candidates
        best_action = np.zeros(self.n_states)
        best_value = np.inf
        action_std = np.sqrt(1.0 / (self.sigma + 1e-8))  # Exploration noise
        
        # Path integral control
        for _ in range(n_candidates):
            # Sample action trajectory
            action = np.random.normal(0, action_std, size=self.n_states)
            total_value = 0
            
            # Simulate forward
            states = self.state.belief_means.copy()
            for t in range(horizon):
                # Compute value for this step
                value = self._compute_expected_free_energy(states, action)
                total_value += value * (0.9**t)  # Discount factor
                
                # Simulate forward
                flow = self._flow_mapping(states, action)
                states += self.dt * flow
            
            # Update best action
            if total_value < best_value:
                best_value = total_value
                best_action = action
                
        # Add exploration noise based on uncertainty
        exploration_noise = np.random.normal(
            0,
            1.0 / (self.sigma * np.sqrt(self.state.belief_precisions[:, 0] + 1e-8))
        )
        best_action += exploration_noise
        
        return best_action
        
    def step(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Take a step in the environment.
        
        Args:
            obs: Observation
            
        Returns:
            Tuple of (action, free energy)
        """
        # Update beliefs
        _, free_energy = self._update_beliefs(obs)
        
        # Select action
        action = self._select_action(obs)
        
        # Update state
        self.state.update(obs, action, free_energy)
        
        return action, free_energy
        
    def save_state(self, path: Union[str, Path]) -> None:
        """Save current state to file."""
        path = Path(path)
        state_dict = {
            'belief_means': self.state.belief_means.tolist(),
            'belief_precisions': self.state.belief_precisions.tolist(),
            'time': float(self.state.time),
            'dt': float(self.state.dt),
            'true_state': None if self.state.true_state is None else self.state.true_state.tolist(),
            'history': {
                'observations': [o.tolist() for o in self.state.history['observations']],
                'actions': [a.tolist() for a in self.state.history['actions']],
                'belief_means': [m.tolist() for m in self.state.history['belief_means']],
                'belief_precisions': [p.tolist() for p in self.state.history['belief_precisions']],
                'free_energy': self.state.history['free_energy'],
                'time': self.state.history['time'],
                'true_state': [s.tolist() for s in self.state.history['true_state']] if self.state.history['true_state'] else []
            }
        }
        
        with open(path, 'w') as f:
            json.dump(state_dict, f, indent=2)
            
    def load_state(self, path: Union[str, Path]) -> None:
        """Load state from file."""
        path = Path(path)
        with open(path, 'r') as f:
            state_dict = json.load(f)
        
        self.state = ContinuousState(
            belief_means=np.array(state_dict['belief_means']),
            belief_precisions=np.array(state_dict['belief_precisions']),
            time=float(state_dict['time']),
            dt=float(state_dict['dt']),
            true_state=None if state_dict['true_state'] is None else np.array(state_dict['true_state'])
        )
        
        self.state.history = {
            'observations': [np.array(o) for o in state_dict['history']['observations']],
            'actions': [np.array(a) for a in state_dict['history']['actions']],
            'belief_means': [np.array(m) for m in state_dict['history']['belief_means']],
            'belief_precisions': [np.array(p) for p in state_dict['history']['belief_precisions']],
            'free_energy': state_dict['history']['free_energy'],
            'time': state_dict['history']['time'],
            'true_state': [np.array(s) for s in state_dict['history']['true_state']] if state_dict['history']['true_state'] else []
        }
        
    def reset(self) -> None:
        """Reset the agent to initial state."""
        self.state = ContinuousState(
            belief_means=np.zeros((self.n_states, self.n_orders)),
            belief_precisions=np.ones((self.n_states, self.n_orders)),
            dt=self.dt
        ) 