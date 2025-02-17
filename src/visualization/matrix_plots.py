"""
Visualization utilities for Active Inference matrices and state spaces.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MatrixPlotter:
    """Utility class for matrix visualization."""
    
    def __init__(self, output_dir: str, style_config: dict):
        """Initialize plotter with output directory and style settings.
        
        Args:
            output_dir: Directory to save plots
            style_config: Visualization style configuration
        """
        self.output_dir = Path(output_dir)
        self.style_config = style_config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style parameters
        plt.style.use(self.style_config.get('theme', 'default'))
        self.figure_size = self.style_config.get('figure_size', (8, 6))
        self.dpi = self.style_config.get('dpi', 100)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    def plot_heatmap(self, 
                     matrix: np.ndarray,
                     ax: Optional[plt.Axes] = None,
                     title: Optional[str] = None,
                     xlabel: Optional[str] = None,
                     ylabel: Optional[str] = None,
                     cmap: Optional[str] = None,
                     **kwargs) -> plt.Figure:
        """Plot matrix as heatmap.
        
        Args:
            matrix: 2D array to plot
            ax: Optional matplotlib axes
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            cmap: Colormap name
            **kwargs: Additional arguments for sns.heatmap
            
        Returns:
            Matplotlib figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        else:
            fig = ax.figure
            
        # Remove save_name from kwargs if present
        save_name = kwargs.pop('save_name', None)
        
        sns.heatmap(matrix, ax=ax, cmap=cmap or self.style_config.get('colormap', 'viridis'), **kwargs)
        
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
            
        # Save if save_name provided
        if save_name:
            self.save_figure(fig, save_name)
            
        return fig
    
    def plot_multi_heatmap(self,
                          tensor: np.ndarray,
                          title: str,
                          xlabel: str,
                          ylabel: str,
                          slice_names: List[str],
                          colormap: str = "Blues",
                          save_name: Optional[str] = None) -> plt.Figure:
        """Plot 3D tensor as multiple heatmaps."""
        n_slices = tensor.shape[0]
        fig, axes = plt.subplots(1, n_slices, figsize=(5*n_slices, 4))
        
        if n_slices == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            sns.heatmap(tensor[i],
                       cmap=colormap,
                       annot=True,
                       fmt='.2f',
                       ax=ax,
                       cbar=True)
            ax.set_title(f"{title} - {slice_names[i]}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        
        plt.tight_layout()
        
        if save_name and self.output_dir:
            fig.savefig(self.output_dir / f"{save_name}.png")
        
        return fig
    
    def plot_bar(self, 
                 values: np.ndarray,
                 ax: Optional[plt.Axes] = None,
                 title: Optional[str] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 **kwargs) -> plt.Figure:
        """Plot vector as bar chart.
        
        Args:
            values: 1D array to plot
            ax: Optional matplotlib axes
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            **kwargs: Additional arguments for plt.bar
            
        Returns:
            Matplotlib figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figure_size)
        else:
            fig = ax.figure
            
        # Remove save_name from kwargs if present
        save_name = kwargs.pop('save_name', None)
        
        ax.bar(range(len(values)), values, **kwargs)
        
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
            
        # Save if save_name provided
        if save_name:
            self.save_figure(fig, save_name)
            
        return fig
    
    def plot_belief_evolution(self,
                            beliefs: np.ndarray,
                            ax: Optional[plt.Axes] = None,
                            state_labels: Optional[list] = None) -> plt.Axes:
        """Plot belief evolution over time.
        
        Args:
            beliefs: Array of shape (time_steps, num_states)
            ax: Optional matplotlib axes
            state_labels: Optional list of state names
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            _, ax = plt.subplots(figsize=self.figure_size)
            
        time_steps = np.arange(beliefs.shape[0])
        
        for state in range(beliefs.shape[1]):
            label = f'State {state}' if state_labels is None else state_labels[state]
            ax.plot(time_steps, beliefs[:, state], label=label)
            
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Belief Probability')
        ax.set_title('Belief Evolution')
        ax.legend()
        
        return ax
    
    def plot_free_energy_landscape(self,
                                 free_energy_func,
                                 belief_range: Tuple[float, float] = (0, 1),
                                 num_points: int = 50,
                                 ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot free energy landscape over belief space.
        
        Args:
            free_energy_func: Function that computes free energy given beliefs
            belief_range: Range of belief values to plot
            num_points: Number of points in each dimension
            ax: Optional matplotlib axes
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig = plt.figure(figsize=self.figure_size)
            ax = fig.add_subplot(111, projection='3d')
            
        # Create belief grid
        x = np.linspace(belief_range[0], belief_range[1], num_points)
        y = np.linspace(belief_range[0], belief_range[1], num_points)
        X, Y = np.meshgrid(x, y)
        
        # Compute free energy at each point
        Z = np.zeros_like(X)
        for i in range(num_points):
            for j in range(num_points):
                beliefs = np.array([X[i,j], Y[i,j], 1 - X[i,j] - Y[i,j]])
                if beliefs.min() >= 0 and beliefs.sum() <= 1 + 1e-10:
                    Z[i,j] = free_energy_func(beliefs)
                else:
                    Z[i,j] = np.nan
                    
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap=self.style_config.get('colormap', 'viridis'))
        plt.colorbar(surf, ax=ax)
        
        ax.set_xlabel('Belief in State 0')
        ax.set_ylabel('Belief in State 1')
        ax.set_zlabel('Expected Free Energy')
        
        return ax
    
    def save_figure(self, fig: plt.Figure, name: str, use_timestamp: bool = False) -> Path:
        """Save figure to output directory.
        
        Args:
            fig: Figure to save
            name: Base name for the file
            use_timestamp: Whether to add timestamp to filename
            
        Returns:
            Path to saved file
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.{self.style_config.get('file_format', 'png')}"
        else:
            filename = f"{name}.{self.style_config.get('file_format', 'png')}"
            
        filepath = self.output_dir / filename
        
        # Save figure
        fig.savefig(filepath, bbox_inches='tight', dpi=self.dpi)
        plt.close(fig)
        
        # Log the save operation
        logger.info(f"Saved visualization to: {filepath.absolute()}")
        logger.info(f"File size: {filepath.stat().st_size} bytes")
        
        return filepath

class StateSpacePlotter:
    """Plotting utilities for state spaces and beliefs."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_belief_evolution(self,
                            beliefs: np.ndarray,
                            title: str,
                            state_labels: List[str],
                            save_name: Optional[str] = None) -> plt.Figure:
        """Plot belief evolution over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        time_steps = range(beliefs.shape[0])
        for i in range(beliefs.shape[1]):
            ax.plot(time_steps, beliefs[:, i],
                   label=state_labels[i],
                   marker='o')
        
        ax.set_title(title)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Belief Probability")
        ax.legend()
        
        if save_name and self.save_dir:
            fig.savefig(self.save_dir / f"{save_name}.png")
        
        return fig
    
    def plot_free_energy_landscape(self,
                                 free_energy: np.ndarray,
                                 title: str,
                                 save_name: Optional[str] = None) -> plt.Figure:
        """Plot free energy landscape."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.arange(free_energy.shape[0])
        y = np.arange(free_energy.shape[1])
        X, Y = np.meshgrid(x, y)
        
        surf = ax.plot_surface(X, Y, free_energy, cmap='viridis')
        fig.colorbar(surf)
        
        ax.set_title(title)
        ax.set_xlabel("State Dimension 1")
        ax.set_ylabel("State Dimension 2")
        ax.set_zlabel("Free Energy")
        
        if save_name and self.save_dir:
            fig.savefig(self.save_dir / f"{save_name}.png")
        
        return fig
    
    def plot_policy_evaluation(self,
                             policy_values: np.ndarray,
                             policy_labels: List[str],
                             title: str,
                             save_name: Optional[str] = None) -> plt.Figure:
        """Plot policy evaluation results."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(range(len(policy_values)), policy_values)
        ax.set_xticks(range(len(policy_values)))
        ax.set_xticklabels(policy_labels, rotation=45)
        
        ax.set_title(title)
        ax.set_xlabel("Policies")
        ax.set_ylabel("Expected Free Energy")
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            fig.savefig(self.save_dir / f"{save_name}.png")
        
        return fig

class NetworkPlotter:
    """Plotting utilities for belief networks and relationships."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_belief_network(self,
                          adjacency: np.ndarray,
                          node_labels: List[str],
                          title: str,
                          save_name: Optional[str] = None) -> plt.Figure:
        """Plot belief network structure."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        heatmap = sns.heatmap(adjacency,
                             cmap="YlOrRd",
                             annot=True,
                             fmt='.2f',
                             xticklabels=node_labels,
                             yticklabels=node_labels,
                             ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel("Nodes")
        ax.set_ylabel("Nodes")
        
        if save_name and self.save_dir:
            fig.savefig(self.save_dir / f"{save_name}.png")
        
        return fig 