"""
Visualization utilities for Continuous Active Inference.

This module provides visualization tools for analyzing and debugging the
continuous-time, continuous-state space active inference agent.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def confidence_ellipse(mean_x, mean_y, std_x, std_y, ax, n_std=3.0, **kwargs):
    """
    Create a plot of the covariance confidence ellipse.
    
    Parameters
    ----------
    mean_x : float
        Mean of x
    mean_y : float
        Mean of y
    std_x : float
        Standard deviation of x
    std_y : float
        Standard deviation of y
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    from matplotlib.patches import Ellipse
    
    # Width and height are "full" widths, not radius
    width = 2 * n_std * std_x
    height = 2 * n_std * std_y
    
    ellip = Ellipse((mean_x, mean_y), width, height, **kwargs)
    ax.add_patch(ellip)
    return ellip

class ContinuousVisualizer:
    """Visualization tools for continuous active inference."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')  # Use default style instead of seaborn
        # Configure style manually
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.color': '#CCCCCC',
            'grid.linestyle': '--',
            'grid.alpha': 0.5,
            'lines.linewidth': 2,
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
        # Create subdirectories
        (self.output_dir / 'belief_evolution').mkdir(exist_ok=True)
        (self.output_dir / 'free_energy').mkdir(exist_ok=True)
        (self.output_dir / 'actions').mkdir(exist_ok=True)
        (self.output_dir / 'phase_space').mkdir(exist_ok=True)
        (self.output_dir / 'summary').mkdir(exist_ok=True)
        (self.output_dir / 'animations').mkdir(exist_ok=True)
        
    def plot_belief_evolution(self,
                           belief_means: List[np.ndarray],
                           belief_precisions: List[np.ndarray],
                           save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot the evolution of beliefs over time."""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot means
        for i in range(belief_means[0].shape[0]):  # For each state dimension
            means = [b[i,0] for b in belief_means]  # Get lowest order
            axes[0].plot(means, label=f'State {i+1}')
        axes[0].set_title('Belief Means Evolution')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Mean')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot precisions
        for i in range(belief_precisions[0].shape[0]):
            precisions = [b[i,0] for b in belief_precisions]  # Get lowest order
            axes[1].plot(precisions, label=f'State {i+1}')
        axes[1].set_title('Belief Precisions Evolution')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Precision')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close(fig)

    def plot_free_energy(self,
                        free_energies: np.ndarray,
                        save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot the evolution of free energy over time."""
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(free_energies)
        ax.set_title('Free Energy Evolution')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Free Energy')
        ax.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close(fig)

    def plot_action_evolution(self,
                            actions: List[np.ndarray],
                            save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot the evolution of actions over time."""
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 6))
        actions_array = np.array(actions)
        for i in range(actions_array.shape[1]):
            ax.plot(actions_array[:,i], label=f'Action {i+1}')
        ax.set_title('Action Evolution')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close(fig)

    def plot_phase_space(self,
                        belief_means: List[np.ndarray],
                        belief_precisions: List[np.ndarray],
                        save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot the phase space trajectory of beliefs."""
        plt.style.use('default')
        if belief_means[0].shape[0] < 2:
            logger.warning("Phase space plot requires at least 2 state dimensions")
            return
            
        fig, ax = plt.subplots(figsize=(10, 10))
        means = np.array([b[:,0] for b in belief_means])  # Get lowest order
        precisions = np.array([b[:,0] for b in belief_precisions])
        
        # Plot trajectory
        ax.plot(means[:,0], means[:,1], 'b-', alpha=0.5, label='Trajectory')
        ax.plot(means[0,0], means[0,1], 'go', label='Start')
        ax.plot(means[-1,0], means[-1,1], 'ro', label='End')
        
        # Plot uncertainty ellipses at intervals
        n_points = len(means)
        for i in range(0, n_points, n_points//5):
            confidence_ellipse(
                means[i,0], means[i,1],
                1/np.sqrt(precisions[i,0]), 1/np.sqrt(precisions[i,1]),
                ax, n_std=2, alpha=0.1
            )
            
        ax.set_title('Phase Space Trajectory')
        ax.set_xlabel('State 1')
        ax.set_ylabel('State 2')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close(fig)

    def create_summary_plot(self,
                          history: Dict,
                          save_path: Optional[Union[str, Path]] = None) -> None:
        """Create a summary plot with multiple subplots."""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Belief evolution
        for i in range(history['belief_means'][0].shape[0]):
            means = [b[i,0] for b in history['belief_means']]
            axes[0,0].plot(history['time'], means, label=f'State {i+1}')
        axes[0,0].set_title('Belief Evolution')
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('Mean')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Free energy
        axes[0,1].plot(history['time'], history['free_energy'])
        axes[0,1].set_title('Free Energy')
        axes[0,1].set_xlabel('Time')
        axes[0,1].set_ylabel('Free Energy')
        axes[0,1].grid(True)
        
        # Actions
        actions = np.array(history['actions'])
        for i in range(actions.shape[1]):
            axes[1,0].plot(history['time'], actions[:,i], label=f'Action {i+1}')
        axes[1,0].set_title('Actions')
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Action')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Phase space
        if history['belief_means'][0].shape[0] >= 2:
            means = np.array([b[:,0] for b in history['belief_means']])
            axes[1,1].plot(means[:,0], means[:,1], 'b-', alpha=0.5)
            axes[1,1].plot(means[0,0], means[0,1], 'go', label='Start')
            axes[1,1].plot(means[-1,0], means[-1,1], 'ro', label='End')
            axes[1,1].set_title('Phase Space')
            axes[1,1].set_xlabel('State 1')
            axes[1,1].set_ylabel('State 2')
            axes[1,1].legend()
            axes[1,1].grid(True)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close(fig)

    def save_animation(self,
                      history: Dict,
                      save_path: Optional[Union[str, Path]] = None,
                      fps: int = 30) -> None:
        """Create and save an animation of the belief evolution."""
        plt.style.use('default')
        if not save_path:
            return
            
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(10, 10))
        means = np.array([b[:,0] for b in history['belief_means']])
        precisions = np.array([b[:,0] for b in history['belief_precisions']])
        
        def animate(i):
            ax.clear()
            # Plot trajectory up to current frame
            ax.plot(means[:i+1,0], means[:i+1,1], 'b-', alpha=0.5)
            # Plot current point
            ax.plot(means[i,0], means[i,1], 'ro')
            # Plot uncertainty ellipse
            confidence_ellipse(
                means[i,0], means[i,1],
                1/np.sqrt(precisions[i,0]), 1/np.sqrt(precisions[i,1]),
                ax, n_std=2, alpha=0.2
            )
            ax.set_title(f'Time step {i}')
            ax.set_xlabel('State 1')
            ax.set_ylabel('State 2')
            ax.grid(True)
            
        anim = animation.FuncAnimation(
            fig, animate, frames=len(means),
            interval=1000/fps, blit=False
        )
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        anim.save(save_path, writer='pillow', fps=fps)
        plt.close(fig)

    def plot_taylor_expansion(self,
                            belief_means: List[np.ndarray],
                            time_points: np.ndarray,
                            prediction_horizon: float = 0.1,
                            save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot Taylor expansion predictions vs actual trajectories.
        
        Args:
            belief_means: List of belief means over time
            time_points: Time points corresponding to beliefs
            prediction_horizon: How far ahead to predict using Taylor expansion
            save_path: Where to save the plot
        """
        plt.style.use('default')
        fig, axes = plt.subplots(belief_means[0].shape[0], 1, 
                                figsize=(12, 4*belief_means[0].shape[0]),
                                squeeze=False)
        
        means = np.array(belief_means)
        dt = time_points[1] - time_points[0]
        
        for state_idx in range(means.shape[1]):
            ax = axes[state_idx, 0]
            
            # Plot actual trajectory
            ax.plot(time_points, means[:, state_idx, 0], 
                   'b-', label='Actual', linewidth=2)
            
            # Plot Taylor predictions at intervals
            n_predictions = 5
            pred_indices = np.linspace(0, len(time_points)-1, n_predictions, 
                                     dtype=int)[:-1]
            
            for t_idx in pred_indices:
                # Get derivatives at this point
                x = means[t_idx, state_idx, 0]  # Position
                v = means[t_idx, state_idx, 1]  # Velocity
                a = means[t_idx, state_idx, 2]  # Acceleration
                
                # Create prediction times
                t_pred = np.linspace(0, prediction_horizon, 20)
                
                # Taylor expansion prediction
                x_pred = x + v*t_pred + 0.5*a*t_pred**2
                
                # Plot prediction
                t_absolute = time_points[t_idx] + t_pred
                ax.plot(t_absolute, x_pred, 'r--', alpha=0.5)
                
                # Mark prediction start point
                ax.plot(time_points[t_idx], x, 'ro', alpha=0.5)
            
            ax.set_title(f'State {state_idx+1} Trajectory with Taylor Predictions')
            ax.set_xlabel('Time')
            ax.set_ylabel('State Value')
            ax.legend(['Actual', 'Taylor Predictions'])
            ax.grid(True)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close(fig)
        
    def plot_generalized_coordinates_relationships(self,
                                                 belief_means: List[np.ndarray],
                                                 time_points: np.ndarray,
                                                 save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot relationships between different orders of generalized coordinates.
        
        Args:
            belief_means: List of belief means over time
            time_points: Time points corresponding to beliefs
            save_path: Where to save the plot
        """
        plt.style.use('default')
        means = np.array(belief_means)
        dt = time_points[1] - time_points[0]
        
        # Create figure with subplots for each state and order relationship
        n_states = means.shape[1]
        n_orders = means.shape[2] - 1  # Number of derivative relationships
        
        fig, axes = plt.subplots(n_states, n_orders, 
                                figsize=(6*n_orders, 4*n_states))
        if n_states == 1:
            axes = axes.reshape(1, -1)
            
        for state_idx in range(n_states):
            for order_idx in range(n_orders):
                ax = axes[state_idx, order_idx]
                
                # Get current order and its derivative
                x = means[:-1, state_idx, order_idx]
                dx = np.diff(means[:, state_idx, order_idx]) / dt
                v = means[:-1, state_idx, order_idx + 1]
                
                # Plot relationship
                ax.scatter(v, dx, alpha=0.5, s=10)
                
                # Plot y=x line for comparison
                lims = [
                    min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1])
                ]
                ax.plot(lims, lims, 'r--', alpha=0.5)
                
                ax.set_title(f'State {state_idx+1}: Order {order_idx} vs {order_idx+1}')
                ax.set_xlabel(f'Order {order_idx+1} Value')
                ax.set_ylabel(f'Order {order_idx} Derivative')
                ax.grid(True)
                
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close(fig)

    def create_generalized_coordinates_summary(self,
                                            history: Dict,
                                            save_dir: Optional[Union[str, Path]] = None) -> None:
        """Create a comprehensive summary of generalized coordinates analysis.
        
        Args:
            history: Dictionary containing simulation history
            save_dir: Directory to save plots
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
        # 1. Plot Taylor expansion predictions
        self.plot_taylor_expansion(
            history['belief_means'],
            np.array(history['time']),
            save_path=save_dir / 'taylor_expansion.png' if save_dir else None
        )
        
        # 2. Plot generalized coordinates relationships
        self.plot_generalized_coordinates_relationships(
            history['belief_means'],
            np.array(history['time']),
            save_path=save_dir / 'generalized_coordinates.png' if save_dir else None
        )
        
        # 3. Create animation of belief evolution with uncertainty
        self.save_animation(
            history,
            save_path=save_dir / 'belief_evolution.gif' if save_dir else None
        )
        
        # 4. Create summary plot
        self.create_summary_plot(
            history,
            save_path=save_dir / 'summary.png' if save_dir else None
        ) 