"""
SimplePOMDP implementation using Active Inference framework.
This module implements a simple POMDP (Partially Observable Markov Decision Process)
using the principles of Active Inference for decision making and belief updating.
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.active_inference.base import ActiveInferenceModel
from src.utils.matrix_utils import (
    ensure_matrix_properties,
    compute_entropy,
    softmax,
    kl_divergence,
    expected_free_energy
)
from src.visualization.matrix_plots import MatrixPlotter

# Make compute_expected_free_energy available at module level
__all__ = ['SimplePOMDP', 'compute_expected_free_energy']

@dataclass
class ModelState:
    """State container for POMDP."""
    current_state: int = 0
    beliefs: np.ndarray = field(default_factory=lambda: np.array([]))
    history: Dict[str, List] = field(default_factory=lambda: {
        'states': [],
        'observations': [],
        'actions': [],
        'beliefs': [],
        'free_energy': [],      # Variational free energy
        'variational_fe': [],   # Sensemaking/perception
        'expected_fe': [],      # List of lists: EFE for each action at each timestep
        'epistemic_value': [],  # Information gain component of EFE
        'pragmatic_value': [],  # Utility component of EFE
        'policy_priors': []     # E matrix evolution
    })
    time_step: int = 0

    def update(self, state: int, observation: int, action: int, beliefs: np.ndarray, 
              variational_fe: float, expected_fe: np.ndarray, policy_prior: np.ndarray,
              epistemic_value: float, pragmatic_value: float):
        """Update state and history."""
        self.current_state = state
        self.beliefs = beliefs
        self.history['states'].append(state)
        self.history['observations'].append(observation)
        self.history['actions'].append(action)
        self.history['beliefs'].append(beliefs.copy())
        self.history['free_energy'].append(variational_fe)
        self.history['variational_fe'].append(variational_fe)
        self.history['expected_fe'].append(expected_fe.copy())
        self.history['epistemic_value'].append(epistemic_value)
        self.history['pragmatic_value'].append(pragmatic_value)
        self.history['policy_priors'].append(policy_prior.copy())
        self.time_step += 1

def compute_expected_free_energy(
    A: np.ndarray,           # Observation model P(o|s)
    B: np.ndarray,           # Transition model P(s'|s,a)
    C: np.ndarray,           # Log preferences ln P(o)
    beliefs: np.ndarray,     # Current state beliefs Q(s)
    action: int             # Action to evaluate
) -> Tuple[float, float, float]:
    """Compute Expected Free Energy for a single action.
    
    Args:
        A: Observation likelihood matrix [n_obs x n_states]
        B: State transition tensor [n_states x n_states x n_actions]
        C: Log preference vector [n_obs]
        beliefs: Current belief state [n_states]
        action: Action index to evaluate
        
    Returns:
        Tuple of (total_EFE, epistemic_value, pragmatic_value) where:
        - total_EFE: Total Expected Free Energy
        - epistemic_value: Information gain (uncertainty reduction)
        - pragmatic_value: Preference satisfaction (utility)
    """
    # Predicted next state distribution
    Qs_a = B[:, :, action] @ beliefs
    
    # Predicted observation distribution
    Qo_a = A @ Qs_a
    
    # Epistemic value (state uncertainty/information gain)
    epistemic = compute_entropy(Qs_a)
    
    # Pragmatic value (preference satisfaction/utility)
    pragmatic = -np.sum(Qo_a * C)  # Negative because C is log preferences
    
    # Total Expected Free Energy
    total_efe = epistemic + pragmatic
    
    return total_efe, epistemic, pragmatic

def update_policy_prior(
    A: np.ndarray,           # Observation model P(o|s)
    B: np.ndarray,           # Transition model P(s'|s,a)
    C: np.ndarray,           # Log preferences ln P(o)
    E: np.ndarray,           # Current policy prior P(a)
    beliefs: np.ndarray,     # Current state beliefs Q(s)
    alpha: float = 0.1,      # Learning rate
    gamma: float = 1.0       # Precision
) -> np.ndarray:
    """Update policy prior using Expected Free Energy.
    
    Args:
        A: Observation likelihood matrix [n_obs x n_states]
        B: State transition tensor [n_states x n_states x n_actions]
        C: Log preference vector [n_obs]
        E: Current policy prior [n_actions]
        beliefs: Current belief state [n_states]
        alpha: Learning rate (0 for static prior)
        gamma: Precision parameter
        
    Returns:
        Updated policy prior E [n_actions]
    """
    n_actions = B.shape[2]
    G = np.zeros(n_actions)
    
    for a in range(n_actions):
        # Get total EFE (first element of tuple)
        G[a], _, _ = compute_expected_free_energy(A, B, C, beliefs, a)
    
    # Compute new policy distribution using softmax
    E_new = softmax(-gamma * G)
    
    # Update with learning rate
    E_updated = (1 - alpha) * E + alpha * E_new
    
    return E_updated

class SimplePOMDP(ActiveInferenceModel):
    """
    A simple POMDP implementation using Active Inference principles.
    
    This class implements a basic POMDP where:
    - States are partially observable
    - Actions influence state transitions
    - Observations provide incomplete information about states
    - Beliefs are updated using Active Inference
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the SimplePOMDP model.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        super().__init__(config_path)
        self.plotter = MatrixPlotter(
            self.config['visualization']['output_dir'],
            self.config['visualization']['style']
        )
        self._initialize_state()
        
    def _load_config(self, config_path: Union[str, Path, Dict]) -> Dict:
        """Load and validate configuration from YAML file or dict.
        
        Args:
            config_path: Path to YAML file or config dictionary
            
        Returns:
            Loaded and validated configuration dictionary
        """
        if isinstance(config_path, (str, Path)):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif isinstance(config_path, dict):
            config = config_path
        else:
            raise TypeError("config_path must be a string, Path, or dictionary")
        
        # Basic validation of required fields
        required_fields = ['model', 'state_space', 'observation_space', 
                         'action_space', 'matrices', 'inference']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in config")
        
        return config
    
    def _initialize_matrices(self):
        """Initialize model matrices."""
        # Initialize A matrix (Observation Model)
        A_init = np.random.rand(
            self.config['observation_space']['num_observations'],
            self.config['state_space']['num_states']
        )
        
        # Apply constraints from config
        constraints = self.config['matrices']['A_matrix'].get('constraints', ['column_stochastic'])
        self.A = ensure_matrix_properties(A_init, constraints)
        
        # Initialize B matrix (Transition/Dynamics)
        self.B = np.zeros((
            self.config['state_space']['num_states'],
            self.config['state_space']['num_states'],
            self.config['action_space']['num_actions']
        ))
        
        for a in range(self.config['action_space']['num_actions']):
            # Initialize based on method specified in config
            init_method = self.config['matrices']['B_matrix']['initialization']
            if init_method == 'identity_based':
                # Get strength parameter (probability of staying in same state)
                strength = self.config['matrices']['B_matrix']['initialization_params']['strength']
                # Create identity matrix with strength on diagonal
                B_a = np.eye(self.config['state_space']['num_states']) * strength
                # Distribute remaining probability uniformly
                off_diag_prob = (1 - strength) / (self.config['state_space']['num_states'] - 1)
                B_a[~np.eye(self.config['state_space']['num_states'], dtype=bool)] = off_diag_prob
            elif init_method == 'uniform':
                # Uniform initialization without normalization
                B_a = np.ones((
                    self.config['state_space']['num_states'],
                    self.config['state_space']['num_states']
                ))
            else:
                # Default to random initialization
                B_a = np.random.rand(
                    self.config['state_space']['num_states'],
                    self.config['state_space']['num_states']
                )
            
            # Apply constraints if specified
            constraints = self.config['matrices']['B_matrix'].get('constraints', [])
            if 'column_stochastic' in constraints:
                B_a = B_a / B_a.sum(axis=0, keepdims=True)
            
            self.B[:, :, a] = B_a
        
        # Initialize C matrix (Log Preferences)
        preferences = np.array(self.config['matrices']['C_matrix']['initialization_params']['preferences'])
        self.C = preferences  # Already in log space
        
        # Initialize D matrix (Prior Beliefs)
        self.D = np.ones(self.config['state_space']['num_states'])
        self.D = self.D / np.sum(self.D)  # Normalize to sum to 1
        
        # Initialize E matrix (Action Prior)
        self.E = np.ones(self.config['action_space']['num_actions'])
        self.E = self.E / np.sum(self.E)  # Initial uniform distribution over actions
        
        # Validate matrix properties
        self._validate_matrices()
    
    def _validate_matrices(self):
        """Validate matrix properties."""
        # Check A matrix (observation model)
        if not np.allclose(self.A.sum(axis=0), 1.0):
            raise ValueError("A matrix must be column stochastic")
        
        # Check B matrix (transition model)
        for a in range(self.B.shape[2]):
            if not np.allclose(self.B[:, :, a].sum(axis=0), 1.0):
                raise ValueError(f"B matrix for action {a} must be column stochastic")
        
        # Check dimensions
        n_states = self.config['state_space']['num_states']
        n_obs = self.config['observation_space']['num_observations']
        n_actions = self.config['action_space']['num_actions']
        
        if self.A.shape != (n_obs, n_states):
            raise ValueError(f"A matrix shape {self.A.shape} does not match (n_obs, n_states) = ({n_obs}, {n_states})")
        
        if self.B.shape != (n_states, n_states, n_actions):
            raise ValueError(f"B matrix shape {self.B.shape} does not match (n_states, n_states, n_actions) = ({n_states}, {n_states}, {n_actions})")
        
        # Check C matrix (log preferences over observations)
        if self.C.shape != (n_obs,):
            raise ValueError(f"C matrix shape {self.C.shape} does not match (n_obs,) = ({n_obs},)")
        
        if self.D.shape != (n_states,):
            raise ValueError(f"D matrix shape {self.D.shape} does not match (n_states,) = ({n_states},)")
        
        # Check E matrix (action distribution)
        if self.E.shape != (n_actions,):
            raise ValueError(f"E matrix shape {self.E.shape} does not match (n_actions,) = ({n_actions},)")
        if not np.allclose(self.E.sum(), 1.0):
            raise ValueError("E matrix must be normalized (sum to 1)")
        if not np.all(self.E >= 0):
            raise ValueError("E matrix must be non-negative")
    
    def _initialize_state(self):
        """Initialize the model's state and beliefs."""
        self.state = ModelState(
            current_state=self.config['state_space']['initial_state'],
            beliefs=self.D.copy()
        )
    
    def step(self, action: Optional[int] = None) -> Tuple[int, float]:
        """Take a step in the environment.
        
        Args:
            action: Optional action to take. If None, action will be selected using
                   active inference.
                   
        Returns:
            Tuple of (observation, variational free energy)
        """
        if action is None:
            action, expected_fe = self._select_action()
        else:
            # Compute EFE even for provided action
            expected_fe = np.zeros(self.config['action_space']['num_actions'])
            epistemic_values = np.zeros_like(expected_fe)
            pragmatic_values = np.zeros_like(expected_fe)
            
            for a in range(self.config['action_space']['num_actions']):
                efe, epist, prag = compute_expected_free_energy(
                    A=self.A, B=self.B, C=self.C, beliefs=self.state.beliefs, action=a
                )
                expected_fe[a] = efe
                epistemic_values[a] = epist
                pragmatic_values[a] = prag
        
        # Get next state using transition model
        next_state = self._get_next_state(action)
        
        # Get observation from new state
        observation = self._get_observation(next_state)
        
        # Update beliefs and compute variational free energy
        variational_fe = self._update_beliefs(observation, action)
        
        # Get EFE components for selected action
        _, epistemic, pragmatic = compute_expected_free_energy(
            A=self.A, B=self.B, C=self.C, beliefs=self.state.beliefs, action=action
        )
        
        # Update state and history
        self.state.update(
            state=next_state,
            observation=observation,
            action=action,
            beliefs=self.state.beliefs,
            variational_fe=variational_fe,
            expected_fe=expected_fe,
            policy_prior=self.E,
            epistemic_value=epistemic,
            pragmatic_value=pragmatic
        )
        
        return observation, variational_fe
    
    def _select_action(self) -> Tuple[int, np.ndarray]:
        """Select action using current policy prior."""
        # Compute Expected Free Energy for each action
        n_actions = self.config['action_space']['num_actions']
        G = np.zeros(n_actions)
        
        for a in range(n_actions):
            G[a], _, _ = compute_expected_free_energy(
                A=self.A,
                B=self.B,
                C=self.C,
                beliefs=self.state.beliefs,
                action=a
            )
        
        # Update policy prior using Expected Free Energy
        self.E = update_policy_prior(
            A=self.A,
            B=self.B,
            C=self.C,
            E=self.E,
            beliefs=self.state.beliefs,
            alpha=self.config['inference']['policy_learning_rate'],
            gamma=self.config['inference']['temperature']
        )
        
        # Sample action from policy distribution
        action = np.random.choice(len(self.E), p=self.E)
        return action, G
    
    def _update_beliefs(self, observation: int, action: int) -> float:
        """
        Update beliefs using Active Inference and compute variational free energy.
        
        Args:
            observation: Observed state index
            action: Taken action index
            
        Returns:
            Computed variational free energy
        """
        # Prediction error (likelihood)
        likelihood = self.A[observation, :]
        
        # Prior from previous beliefs and transition
        prior = self.B[:, :, action] @ self.state.beliefs
        
        # Posterior beliefs using Bayes rule with learning rate
        lr = self.config['inference']['learning_rate']
        posterior = likelihood * prior
        posterior = posterior / posterior.sum()  # Normalize
        
        # Apply learning rate
        posterior = (1 - lr) * self.state.beliefs + lr * posterior
        
        # Compute variational free energy
        variational_fe = -np.log(likelihood @ prior)
        
        # Update beliefs
        self.state.beliefs = posterior
        
        return variational_fe
    
    def _update_history(self, observation: int, action: int, variational_fe: float, expected_fe: np.ndarray):
        """Update the model's history with current step information."""
        self.state.history['states'].append(self.state.current_state)
        self.state.history['observations'].append(observation)
        self.state.history['actions'].append(action)
        self.state.history['beliefs'].append(self.state.beliefs.copy())
        self.state.history['variational_fe'].append(variational_fe)
        self.state.history['expected_fe'].append(expected_fe.copy())
        self.state.history['policy_priors'].append(self.E.copy())
    
    def visualize(self, plot_type: str, **kwargs):
        """Generate visualizations."""
        if plot_type == "belief_evolution":
            return self._plot_belief_evolution(**kwargs)
        elif plot_type == "free_energy_landscape":
            return self._plot_free_energy_landscape(**kwargs)
        elif plot_type == "policy_evaluation":
            return self._plot_policy_evaluation(**kwargs)
        elif plot_type == "state_transitions":
            return self._plot_state_transitions(**kwargs)
        elif plot_type == "observation_likelihood":
            return self._plot_observation_likelihood(**kwargs)
        elif plot_type == "belief_history":
            return self._plot_belief_history(**kwargs)
        elif plot_type == "action_history":
            return self._plot_action_history(**kwargs)
        elif plot_type == "free_energies":
            return self._plot_free_energies(**kwargs)
        elif plot_type == "policy_evolution":
            return self._plot_policy_evolution(**kwargs)
        elif plot_type == "efe_components":
            return self._plot_efe_components(**kwargs)
        elif plot_type == "efe_components_detailed":
            return self._plot_efe_components_detailed(**kwargs)
        elif plot_type == "action_distribution":
            return self._plot_action_distribution(**kwargs)
        elif plot_type == "temperature_effects":
            return self._plot_temperature_effects(**kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    
    def _plot_belief_evolution(self, save: bool = True):
        """Plot the evolution of beliefs over time."""
        beliefs = np.array(self.state.history['beliefs'])
        fig, ax = plt.subplots(figsize=self.config['visualization']['style']['figure_size'])
        
        for i in range(beliefs.shape[1]):
            ax.plot(beliefs[:, i], label=f'State {i}')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Belief Probability')
        ax.set_title('Belief Evolution')
        ax.legend()
        
        if save:
            self.plotter.save_figure(fig, 'belief_evolution')
        
        return fig
    
    def _plot_free_energy_landscape(self, save: bool = True):
        """Plot the free energy landscape."""
        # Create a grid of belief points
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, y)
        
        # Compute free energy components for each point
        Z_total = np.zeros_like(X)
        Z_epistemic = np.zeros_like(X)
        Z_pragmatic = np.zeros_like(X)
        
        for i in range(len(x)):
            for j in range(len(y)):
                beliefs = np.array([X[i,j], Y[i,j], 1 - X[i,j] - Y[i,j]])
                if beliefs.min() >= 0:  # Only valid probability distributions
                    total, epist, prag = compute_expected_free_energy(
                        A=self.A,
                        B=self.B,
                        C=self.C,
                        beliefs=beliefs,
                        action=0
                    )
                    Z_total[i,j] = total
                    Z_epistemic[i,j] = epist
                    Z_pragmatic[i,j] = prag
        
        # Create figure with three subplots
        fig = plt.figure(figsize=(15, 5))
        
        # Plot total EFE
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_total, cmap=self.config['visualization']['style']['colormap_3d'])
        ax1.set_title('Total Expected Free Energy')
        ax1.set_xlabel('Belief in State 0')
        ax1.set_ylabel('Belief in State 1')
        ax1.set_zlabel('Value')
        fig.colorbar(surf1, ax=ax1)
        
        # Plot epistemic value
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_epistemic, cmap='Blues')
        ax2.set_title('Epistemic Value\n(Information Gain)')
        ax2.set_xlabel('Belief in State 0')
        ax2.set_ylabel('Belief in State 1')
        ax2.set_zlabel('Value')
        fig.colorbar(surf2, ax=ax2)
        
        # Plot pragmatic value
        ax3 = fig.add_subplot(133, projection='3d')
        surf3 = ax3.plot_surface(X, Y, Z_pragmatic, cmap='Greens')
        ax3.set_title('Pragmatic Value\n(Utility)')
        ax3.set_xlabel('Belief in State 0')
        ax3.set_ylabel('Belief in State 1')
        ax3.set_zlabel('Value')
        fig.colorbar(surf3, ax=ax3)
        
        plt.tight_layout()
        
        if save:
            self.plotter.save_figure(fig, 'free_energy_landscape')
        
        return fig
    
    def _plot_policy_evaluation(self, save: bool = True):
        """Plot the evaluation of different policies."""
        policies = self.E
        expected_free_energies = []
        epistemic_values = []
        pragmatic_values = []
        
        for a in range(len(policies)):
            efe, epist, prag = compute_expected_free_energy(
                A=self.A,
                B=self.B,
                C=self.C,
                beliefs=self.state.beliefs,
                action=a
            )
            expected_free_energies.append(efe)
            epistemic_values.append(epist)
            pragmatic_values.append(prag)
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot total EFE
        ax1.bar(range(len(policies)), expected_free_energies)
        ax1.set_title('Total Expected Free Energy')
        ax1.set_xlabel('Policy Index')
        ax1.set_ylabel('Value')
        
        # Plot epistemic value
        ax2.bar(range(len(policies)), epistemic_values, color='blue')
        ax2.set_title('Epistemic Value (Information Gain)')
        ax2.set_xlabel('Policy Index')
        ax2.set_ylabel('Value')
        
        # Plot pragmatic value
        ax3.bar(range(len(policies)), pragmatic_values, color='green')
        ax3.set_title('Pragmatic Value (Utility)')
        ax3.set_xlabel('Policy Index')
        ax3.set_ylabel('Value')
        
        plt.tight_layout()
        
        if save:
            self.plotter.save_figure(fig, 'policy_evaluation')
        
        return fig
    
    def _plot_state_transitions(self, save: bool = True):
        """Plot the state transition matrices for each action."""
        num_actions = self.B.shape[-1]
        fig, axes = plt.subplots(1, num_actions, 
                               figsize=(self.config['visualization']['style']['figure_size'][0] * num_actions,
                                      self.config['visualization']['style']['figure_size'][1]))
        
        if num_actions == 1:
            axes = [axes]
        
        for action in range(num_actions):
            self.plotter.plot_heatmap(
                self.B[:,:,action],
                ax=axes[action],
                title=f'State Transitions (Action {action})',
                xlabel='Current State',
                ylabel='Next State'
            )
        
        if save:
            self.plotter.save_figure(fig, 'state_transitions')
        
        return fig
    
    def _plot_observation_likelihood(self, save: bool = True):
        """Plot the observation likelihood matrix."""
        fig, ax = plt.subplots(figsize=self.config['visualization']['style']['figure_size'])
        
        self.plotter.plot_heatmap(
            self.A,
            ax=ax,
            title='Observation Likelihood',
            xlabel='State',
            ylabel='Observation'
        )
        
        if save:
            self.plotter.save_figure(fig, 'observation_likelihood')
        
        return fig
    
    def _plot_belief_history(self, save: bool = True) -> plt.Figure:
        """Plot complete history of belief evolution as heatmap."""
        beliefs = np.array(self.state.history['beliefs'])
        
        fig, ax = plt.subplots(figsize=self.config['visualization']['style']['figure_size'])
        
        sns.heatmap(beliefs.T,
                   cmap='YlOrRd',
                   xticklabels=range(beliefs.shape[0]),
                   yticklabels=[f'State {i}' for i in range(beliefs.shape[1])],
                   ax=ax)
        
        ax.set_title('Belief History Heatmap')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('State')
        
        if save:
            self.plotter.save_figure(fig, 'belief_history')
        
        return fig

    def _plot_action_history(self, save: bool = True) -> plt.Figure:
        """Plot history of selected actions."""
        actions = self.state.history['actions']
        
        fig, ax = plt.subplots(figsize=self.config['visualization']['style']['figure_size'])
        
        # Create scatter plot with jittered y-values for better visibility
        y_jitter = np.random.normal(0, 0.1, len(actions))
        ax.scatter(range(len(actions)), np.array(actions) + y_jitter,
                   c=range(len(actions)), cmap='viridis')
        
        ax.set_title('Action History')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action')
        ax.set_yticks(range(self.config['action_space']['num_actions']))
        ax.set_yticklabels(self.config['action_space']['action_labels'])
        
        if save:
            self.plotter.save_figure(fig, 'action_history')
        
        return fig

    def _plot_free_energies(self, save: bool = True) -> plt.Figure:
        """Plot both Variational and Expected Free Energies over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot Variational Free Energy (perception)
        vfe = self.state.history['variational_fe']
        ax1.plot(vfe, 'b-', label='VFE')
        ax1.set_title('Variational Free Energy (Perception/Sensemaking)')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Free Energy')
        ax1.grid(True)
        
        # Add trend line
        z = np.polyfit(range(len(vfe)), vfe, 1)
        p = np.poly1d(z)
        ax1.plot(range(len(vfe)), p(range(len(vfe))),
                linestyle='--', color='r', label='Trend')
        ax1.legend()
        
        # Plot Expected Free Energy for each action
        efe = np.array(self.state.history['expected_fe'])
        for a in range(efe.shape[1]):
            ax2.plot(efe[:, a], 
                    label=f'Action: {self.config["action_space"]["action_labels"][a]}',
                    marker='o', markersize=4)
        
        ax2.set_title('Expected Free Energy per Action')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Expected Free Energy')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            self.plotter.save_figure(fig, 'free_energies')
        
        return fig

    def _plot_policy_evolution(self, save: bool = True) -> plt.Figure:
        """Plot evolution of policy prior (E matrix) over time."""
        policy_priors = np.array(self.state.history['policy_priors'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for a in range(policy_priors.shape[1]):
            ax.plot(policy_priors[:, a],
                    label=f'Action: {self.config["action_space"]["action_labels"][a]}',
                    marker='o', markersize=4)
        
        ax.set_title('Policy Prior Evolution')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Prior Probability')
        ax.grid(True)
        ax.legend()
        
        if save:
            self.plotter.save_figure(fig, 'policy_evolution')
        
        return fig

    def _plot_efe_components(self, save: bool = True) -> plt.Figure:
        """Plot the epistemic and pragmatic components of Expected Free Energy over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot epistemic value (information gain)
        epistemic = self.state.history['epistemic_value']
        ax1.plot(epistemic, 'b-', label='Epistemic Value')
        ax1.set_title('Epistemic Value (Information Gain)')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Value')
        ax1.grid(True)
        
        # Add trend line for epistemic value
        if len(epistemic) > 1:
            z = np.polyfit(range(len(epistemic)), epistemic, 1)
            p = np.poly1d(z)
            ax1.plot(range(len(epistemic)), p(range(len(epistemic))),
                    linestyle='--', color='r', label='Trend')
        ax1.legend()
        
        # Plot pragmatic value (utility)
        pragmatic = self.state.history['pragmatic_value']
        ax2.plot(pragmatic, 'g-', label='Pragmatic Value')
        ax2.set_title('Pragmatic Value (Utility)')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Value')
        ax2.grid(True)
        
        # Add trend line for pragmatic value
        if len(pragmatic) > 1:
            z = np.polyfit(range(len(pragmatic)), pragmatic, 1)
            p = np.poly1d(z)
            ax2.plot(range(len(pragmatic)), p(range(len(pragmatic))),
                    linestyle='--', color='r', label='Trend')
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            self.plotter.save_figure(fig, 'efe_components')
        
        return fig

    def _plot_efe_components_detailed(self, save: bool = True) -> plt.Figure:
        """Plot detailed breakdown of Expected Free Energy components.
        
        This visualization shows:
        1. Total EFE over time
        2. Epistemic (Information Gain) component
        3. Pragmatic (Utility) component
        4. Component ratio and relationship
        5. Running averages to show trends
        """
        # Create figure with GridSpec for flexible layout
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1])
        
        # Get history data
        epistemic = np.array(self.state.history['epistemic_value'])
        pragmatic = np.array(self.state.history['pragmatic_value'])
        total_efe = epistemic + pragmatic
        time_steps = np.arange(len(epistemic))
        
        # 1. Total EFE Plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_steps, total_efe, 'k-', label='Total EFE')
        ax1.set_title('Total Expected Free Energy')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Value')
        ax1.grid(True)
        
        # Add trend line
        z = np.polyfit(time_steps, total_efe, 1)
        p = np.poly1d(z)
        ax1.plot(time_steps, p(time_steps), 'r--', label='Trend')
        ax1.legend()
        
        # 2. Components Stacked Area Plot (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.fill_between(time_steps, 0, epistemic, alpha=0.5, label='Epistemic (Information Gain)',
                        color='blue')
        ax2.fill_between(time_steps, epistemic, epistemic + pragmatic, alpha=0.5,
                        label='Pragmatic (Utility)', color='green')
        ax2.set_title('EFE Components (Stacked)')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Component Ratio Plot (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        ratio = np.abs(epistemic) / (np.abs(epistemic) + np.abs(pragmatic) + 1e-10)
        ax3.plot(time_steps, ratio, 'b-', label='Epistemic Ratio')
        ax3.plot(time_steps, 1 - ratio, 'g-', label='Pragmatic Ratio')
        ax3.set_title('Component Ratio')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Ratio')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Component Scatter Plot (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        scatter = ax4.scatter(epistemic, pragmatic, c=time_steps, cmap='viridis')
        ax4.set_title('Epistemic vs Pragmatic Value')
        ax4.set_xlabel('Epistemic Value (Information Gain)')
        ax4.set_ylabel('Pragmatic Value (Utility)')
        plt.colorbar(scatter, ax=ax4, label='Time Step')
        ax4.grid(True)
        
        # 5. Running Averages (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        window = min(5, len(time_steps))  # 5-step running average
        if window > 1:
            running_avg_epistemic = np.convolve(epistemic, np.ones(window)/window, mode='valid')
            running_avg_pragmatic = np.convolve(pragmatic, np.ones(window)/window, mode='valid')
            running_avg_total = np.convolve(total_efe, np.ones(window)/window, mode='valid')
            valid_steps = time_steps[window-1:]
            
            ax5.plot(valid_steps, running_avg_epistemic, 'b-', label='Epistemic (Avg)')
            ax5.plot(valid_steps, running_avg_pragmatic, 'g-', label='Pragmatic (Avg)')
            ax5.plot(valid_steps, running_avg_total, 'k-', label='Total EFE (Avg)')
        ax5.set_title(f'Running Averages (Window={window})')
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Value')
        ax5.legend()
        ax5.grid(True)
        
        # Add formula explanation as text
        fig.text(0.02, 0.02, r"""
$G(\pi) = \text{Epistemic}(H[Q(s|\pi)]) + \text{Pragmatic}(D_{KL}[Q(o|\pi)||P(o)])$
""", fontsize=10)
        
        plt.tight_layout()
        
        if save:
            self.plotter.save_figure(fig, 'efe_components_detailed')
        
        return fig

    def _plot_action_distribution(self, save: bool = True):
        """Plot the action probability distribution."""
        fig, ax = plt.subplots(figsize=self.config['visualization']['style']['figure_size'])
        
        # Plot action probabilities
        sns.barplot(
            x=self.config['action_space']['action_labels'],
            y=self.E,
            ax=ax
        )
        ax.set_title('Action Probability Distribution')
        ax.set_xlabel('Actions')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        
        if save:
            self.plotter.save_figure(fig, 'action_distribution')
        
        return fig

    def _plot_temperature_effects(self, G: np.ndarray, temperatures: List[float], save: bool = True):
        """Plot effect of temperature on action distribution."""
        fig, ax = plt.subplots(figsize=self.config['visualization']['style']['figure_size'])
        
        for temp in temperatures:
            E = softmax(-temp * G)
            ax.plot(
                self.config['action_space']['action_labels'],
                E,
                label=f'T={temp}'
            )
        
        ax.set_title('Temperature Effects on Action Distribution')
        ax.set_xlabel('Actions')
        ax.set_ylabel('Probability')
        ax.legend()
        
        if save:
            self.plotter.save_figure(fig, 'temperature_effects')
        
        return fig

    def update_action_distribution(self, expected_free_energy: np.ndarray, temperature: float = 1.0):
        """Update action distribution based on Expected Free Energy.
        
        Args:
            expected_free_energy: Expected Free Energy for each action
            temperature: Temperature parameter for softmax
        """
        self.E = softmax(-temperature * expected_free_energy)
        # Validate updated distribution
        if not np.allclose(self.E.sum(), 1.0):
            raise ValueError("Updated E matrix must be normalized")
        if not np.all(self.E >= 0):
            raise ValueError("Updated E matrix must be non-negative")
        
        # Log entropy for analysis
        entropy = -np.sum(self.E * np.log(self.E + 1e-10))
        self.history['action_entropy'].append(entropy)

    def sample_action(self) -> int:
        """Sample action from current distribution.
        
        Returns:
            Selected action index
        """
        return np.random.choice(len(self.E), p=self.E)

    def _get_next_state(self, action: int) -> int:
        """Get next state using transition model.
        
        Args:
            action: Action index
            
        Returns:
            Next state index
        """
        # Ensure action is valid
        if not 0 <= action < self.config['action_space']['num_actions']:
            raise ValueError(f"Invalid action {action}. Must be between 0 and {self.config['action_space']['num_actions']-1}")
        
        # Get transition probabilities for current state and action
        state_probs = self.B[:, self.state.current_state, action]
        # Ensure probabilities sum to 1
        state_probs = state_probs / state_probs.sum()
        
        # Sample next state
        next_state = np.random.choice(len(state_probs), p=state_probs)
        return next_state
    
    def _get_observation(self, state: int) -> int:
        """Get observation from state.
        
        Args:
            state: State index
            
        Returns:
            Observation index
        """
        # Get observation probabilities for current state
        obs_probs = self.A[:, state]
        # Ensure probabilities sum to 1
        obs_probs = obs_probs / obs_probs.sum()
        
        # Sample observation
        observation = np.random.choice(len(obs_probs), p=obs_probs)
        return observation 