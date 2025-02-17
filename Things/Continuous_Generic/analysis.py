"""
Analysis utilities for Continuous Active Inference.

This module provides tools for analyzing and visualizing the relationship between
generalized coordinates, Taylor series expansions, and belief updating in the
continuous-time active inference framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from pathlib import Path
from typing import List, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContinuousAnalyzer:
    """Analysis tools for continuous active inference."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """Initialize analyzer.
        
        Args:
            output_dir: Directory to save analysis plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create analysis subdirectories
        (self.output_dir / 'taylor_series').mkdir(exist_ok=True)
        (self.output_dir / 'generalized_coords').mkdir(exist_ok=True)
        (self.output_dir / 'belief_dynamics').mkdir(exist_ok=True)
        (self.output_dir / 'prediction_errors').mkdir(exist_ok=True)
        
    def plot_taylor_expansion(self,
                            states: np.ndarray,
                            time_points: np.ndarray,
                            orders: List[int],
                            save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot Taylor series expansion of state trajectory.
        
        Args:
            states: State trajectory in generalized coordinates [n_states, n_orders]
            time_points: Time points for expansion
            orders: List of orders to include in expansion
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot actual trajectory
        plt.plot(time_points, states[:, 0], 'k-', label='Actual', linewidth=2)
        
        # Plot Taylor expansions of different orders
        colors = plt.cm.viridis(np.linspace(0, 1, len(orders)))
        t0 = time_points[0]
        x0 = states[0]
        
        for order, color in zip(orders, colors):
            # Compute Taylor expansion
            expansion = np.zeros_like(time_points)
            for n in range(order + 1):
                expansion += (x0[n] / factorial(n)) * (time_points - t0)**n
            
            plt.plot(time_points, expansion, '--', 
                    color=color, 
                    label=f'Order {order}',
                    alpha=0.7)
        
        plt.title('Taylor Series Expansion of State Trajectory')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_generalized_coordinates(self,
                                   states: np.ndarray,
                                   time_points: np.ndarray,
                                   save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot state representation in generalized coordinates.
        
        Args:
            states: State trajectory in generalized coordinates [n_states, n_orders]
            time_points: Time points
            save_path: Path to save plot
        """
        n_orders = states.shape[1]
        
        fig, axes = plt.subplots(n_orders, 1, figsize=(12, 4*n_orders))
        if n_orders == 1:
            axes = [axes]
            
        for i, ax in enumerate(axes):
            ax.plot(time_points, states[:, i], 'b-', linewidth=2)
            ax.set_title(f'Order {i} (d^{i}x/dt^{i})')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_belief_dynamics(self,
                           belief_means: List[np.ndarray],
                           belief_precisions: List[np.ndarray],
                           time_points: np.ndarray,
                           save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot belief dynamics in phase space with uncertainty.
        
        Args:
            belief_means: List of belief means [n_states, n_orders]
            belief_precisions: List of belief precisions [n_states, n_orders]
            time_points: Time points
            save_path: Path to save plot
        """
        from matplotlib.patches import Ellipse
        
        plt.figure(figsize=(12, 12))
        
        # Convert to arrays
        means = np.array(belief_means)
        precisions = np.array(belief_precisions)
        
        # Plot trajectory
        plt.plot(means[:, 0, 0], means[:, 1, 0], 'b-', alpha=0.5, label='Trajectory')
        
        # Plot uncertainty ellipses at intervals
        n_points = len(time_points)
        for i in range(0, n_points, n_points//5):
            mean = means[i, :, 0]
            prec = precisions[i, :, 0]
            
            # Create covariance ellipse
            std = 1.0 / np.sqrt(prec + 1e-8)
            ellip = Ellipse(mean, width=2*std[0], height=2*std[1],
                          alpha=0.2, fc='gray', ec='none')
            plt.gca().add_patch(ellip)
            
            # Add time label
            plt.annotate(f't={time_points[i]:.2f}', 
                        xy=mean, 
                        xytext=(10, 10),
                        textcoords='offset points')
        
        plt.title('Belief Dynamics in Phase Space')
        plt.xlabel('State 1')
        plt.ylabel('State 2')
        plt.axis('equal')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_prediction_errors(self,
                             observations: np.ndarray,
                             predictions: np.ndarray,
                             time_points: np.ndarray,
                             save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot prediction errors across time.
        
        Args:
            observations: Actual observations [n_timesteps, n_obs]
            predictions: Predicted observations [n_timesteps, n_obs]
            time_points: Time points
            save_path: Path to save plot
        """
        n_obs = observations.shape[1]
        
        fig, axes = plt.subplots(n_obs, 1, figsize=(12, 4*n_obs))
        if n_obs == 1:
            axes = [axes]
            
        for i, ax in enumerate(axes):
            # Plot actual and predicted
            ax.plot(time_points, observations[:, i], 'b-', label='Actual', alpha=0.7)
            ax.plot(time_points, predictions[:, i], 'r--', label='Predicted', alpha=0.7)
            
            # Plot error
            error = observations[:, i] - predictions[:, i]
            ax.fill_between(time_points, 0, error, 
                          color='gray', alpha=0.2, label='Error')
            
            ax.set_title(f'Observation {i+1}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def analyze_convergence(self,
                          free_energies: np.ndarray,
                          time_points: np.ndarray,
                          window_size: int = 10,
                          save_path: Optional[Union[str, Path]] = None) -> None:
        """Analyze convergence of free energy minimization.
        
        Args:
            free_energies: Free energy values over time
            time_points: Time points
            window_size: Window size for moving statistics
            save_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot raw free energy
        ax1.plot(time_points, free_energies, 'b-', alpha=0.7)
        ax1.set_title('Free Energy Evolution')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Free Energy')
        ax1.grid(True)
        
        # Compute and plot convergence metrics
        dF = np.diff(free_energies)
        conv_rate = np.zeros_like(free_energies[:-window_size])
        
        for i in range(len(conv_rate)):
            conv_rate[i] = np.mean(np.abs(dF[i:i+window_size]))
            
        ax2.plot(time_points[:-window_size], conv_rate, 'r-', alpha=0.7)
        ax2.set_title('Convergence Rate (Moving Average of |dF/dt|)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Rate')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close() 