"""
Visualization module for Generic POMDP implementation.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml

class POMDPVisualizer:
    """Visualizer for Generic POMDP."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize visualizer.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_style()
        
    def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict:
        """Load visualization configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if config_path is None:
            config_path = Path(__file__).parent / 'configuration.yaml'
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config['visualization']
        
    def _setup_style(self):
        """Setup plotting style."""
        plt.style.use('default')  # Use default style instead of seaborn
        plt.rcParams.update({
            'figure.figsize': self.config['style']['figure_size'],
            'font.size': self.config['style']['font_size']
        })
        
    def _setup_output_dir(self) -> Path:
        """Setup output directory.
        
        Returns:
            Path to output directory
        """
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
        
    def plot_belief_evolution(self,
                            beliefs: List[np.ndarray],
                            save: bool = True) -> None:
        """Plot evolution of beliefs over time.
        
        Args:
            beliefs: List of belief arrays
            save: Whether to save the plot
        """
        plt.figure()
        
        # Convert to array for easier plotting
        beliefs_array = np.array(beliefs)
        
        # Plot each state's belief trajectory
        for s in range(beliefs_array.shape[1]):
            plt.plot(beliefs_array[:,s],
                    label=f'State {s}',
                    alpha=0.8)
            
        plt.xlabel('Time Step')
        plt.ylabel('Belief Probability')
        plt.title('Belief Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            self._save_plot('belief_evolution')
        
    def plot_free_energy(self,
                        free_energies: List[float],
                        save: bool = True) -> None:
        """Plot free energy over time.
        
        Args:
            free_energies: List of free energy values
            save: Whether to save the plot
        """
        plt.figure()
        
        plt.plot(free_energies, 'b-', alpha=0.8)
        plt.xlabel('Time Step')
        plt.ylabel('Free Energy')
        plt.title('Free Energy Evolution')
        plt.grid(True, alpha=0.3)
        
        if save:
            self._save_plot('free_energy')
            
    def plot_action_probabilities(self,
                                action_probs: List[np.ndarray],
                                save: bool = True) -> None:
        """Plot action probability evolution.
        
        Args:
            action_probs: List of action probability arrays
            save: Whether to save the plot
        """
        plt.figure()
        
        # Convert to array for easier plotting
        probs_array = np.array(action_probs)
        
        # Plot each action's probability trajectory
        for a in range(probs_array.shape[1]):
            plt.plot(probs_array[:,a],
                    label=f'Action {a}',
                    alpha=0.8)
            
        plt.xlabel('Time Step')
        plt.ylabel('Action Probability')
        plt.title('Action Selection Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            self._save_plot('action_probabilities')
            
    def plot_observation_counts(self,
                              observations: List[int],
                              num_observations: int,
                              save: bool = True) -> None:
        """Plot histogram of observations.
        
        Args:
            observations: List of observation indices
            num_observations: Total number of possible observations
            save: Whether to save the plot
        """
        plt.figure()
        
        plt.hist(observations,
                bins=np.arange(num_observations + 1) - 0.5,
                rwidth=0.8,
                alpha=0.8)
        plt.xlabel('Observation')
        plt.ylabel('Count')
        plt.title('Observation Distribution')
        plt.grid(True, alpha=0.3)
        
        if save:
            self._save_plot('observation_counts')
            
    def plot_state_transition_matrix(self,
                                   B_matrix: np.ndarray,
                                   action: int,
                                   save: bool = True) -> None:
        """Plot state transition matrix for given action.
        
        Args:
            B_matrix: State transition matrix
            action: Action index
            save: Whether to save the plot
        """
        plt.figure()
        
        plt.imshow(B_matrix[:,:,action],
                  cmap=self.config['style']['colormap'],
                  aspect='auto')
        plt.colorbar(label='Transition Probability')
        plt.xlabel('Current State')
        plt.ylabel('Next State')
        plt.title(f'State Transition Matrix (Action {action})')
        
        if save:
            self._save_plot(f'transition_matrix_action_{action}')
            
    def plot_observation_matrix(self,
                              A_matrix: np.ndarray,
                              save: bool = True) -> None:
        """Plot observation matrix.
        
        Args:
            A_matrix: Observation matrix
            save: Whether to save the plot
        """
        plt.figure()
        
        plt.imshow(A_matrix,
                  cmap=self.config['style']['colormap'],
                  aspect='auto')
        plt.colorbar(label='Observation Probability')
        plt.xlabel('State')
        plt.ylabel('Observation')
        plt.title('Observation Matrix')
        
        if save:
            self._save_plot('observation_matrix')
            
    def plot_preferences(self,
                        C_matrix: np.ndarray,
                        save: bool = True) -> None:
        """Plot preference matrix.
        
        Args:
            C_matrix: Preference matrix
            save: Whether to save the plot
        """
        plt.figure()
        
        plt.imshow(C_matrix,
                  cmap=self.config['style']['colormap'],
                  aspect='auto')
        plt.colorbar(label='Preference Value')
        plt.xlabel('Time Step')
        plt.ylabel('Observation')
        plt.title('Preference Matrix')
        
        if save:
            self._save_plot('preferences')
            
    def plot_belief_entropy(self,
                           beliefs: List[np.ndarray],
                           save: bool = True) -> None:
        """Plot belief entropy over time.
        
        Args:
            beliefs: List of belief arrays
            save: Whether to save the plot
        """
        plt.figure()
        
        # Compute entropy for each belief state
        entropies = [-np.sum(b * np.log2(b + 1e-12)) for b in beliefs]
        
        plt.plot(entropies, 'r-', alpha=0.8)
        plt.xlabel('Time Step')
        plt.ylabel('Belief Entropy (bits)')
        plt.title('Belief Entropy Evolution')
        plt.grid(True, alpha=0.3)
        
        if save:
            self._save_plot('belief_entropy')
            
    def plot_all(self,
                 model_state: Dict,
                 model_params: Dict) -> None:
        """Plot all available visualizations.
        
        Args:
            model_state: Dictionary containing model state history
            model_params: Dictionary containing model parameters
        """
        # Plot belief evolution
        self.plot_belief_evolution(model_state['history']['beliefs'])
        
        # Plot free energy
        self.plot_free_energy(model_state['history']['free_energy'])
        
        # Plot action probabilities if available
        if 'policy_probs' in model_state['history']:
            self.plot_action_probabilities(model_state['history']['policy_probs'])
        
        # Plot observation counts
        self.plot_observation_counts(
            model_state['history']['observations'],
            model_params['num_observations']
        )
        
        # Plot matrices
        self.plot_observation_matrix(model_params['A'])
        
        for a in range(model_params['num_actions']):
            self.plot_state_transition_matrix(model_params['B'], a)
            
        self.plot_preferences(model_params['C'])
        
        # Plot belief entropy
        self.plot_belief_entropy(model_state['history']['beliefs'])
        
    def _save_plot(self, name: str) -> None:
        """Save current plot to file.
        
        Args:
            name: Base name for the plot file
        """
        output_dir = self._setup_output_dir()
        
        # Save in all configured formats
        for fmt in self.config['formats']:
            path = output_dir / f'{name}.{fmt}'
            plt.savefig(path,
                       dpi=self.config['style']['dpi'],
                       bbox_inches='tight') 