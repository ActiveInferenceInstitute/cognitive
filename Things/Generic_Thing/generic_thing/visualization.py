"""Visualization tools for analyzing and plotting free energy landscapes and belief updates.

This module provides visualization capabilities for:
- Free energy landscapes across state spaces
- Message passing network dynamics 
- Markov blanket state relationships
- Belief update trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import logging
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class Visualizer:
    """Class containing visualization methods for analyzing free energy and belief dynamics."""
    
    def __init__(self):
        """Initialize the visualizer with default plot settings."""
        sns.set_style('whitegrid')  # Use seaborn's whitegrid style
        self.default_figsize = (10, 6)
        
        # Set up visualization directory
        self.viz_dir = Path(os.path.dirname(__file__)) / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized visualizer with output directory: {self.viz_dir}")
        
    def _ensure_viz_path(self, filename: Union[str, Path]) -> Path:
        """Ensure visualization output path is within visualization directory.
        
        Args:
            filename: Name or path of the file to save
            
        Returns:
            Path object for the output file within visualization directory
        """
        path = Path(filename)
        
        # If path is just a filename or is outside viz_dir, put it in viz_dir
        if len(path.parts) == 1 or not str(path).startswith(str(self.viz_dir)):
            path = self.viz_dir / path.name
            
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving visualization to: {path}")
        return path
        
    def plot_free_energy_landscape(self, 
                                 internal_states: np.ndarray,
                                 external_states: np.ndarray,
                                 free_energies: np.ndarray,
                                 title: str = "Free Energy Landscape",
                                 save_path: Optional[str] = None) -> None:
        """Plot a 3D surface of the free energy landscape.
        
        Args:
            internal_states: Array of internal state values
            external_states: Array of external state values  
            free_energies: Array of free energy values for each state combination
            title: Plot title
            save_path: Optional path to save the plot
        """
        logger.info("Plotting free energy landscape")
        fig = plt.figure(figsize=self.default_figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(internal_states, external_states)
        surf = ax.plot_surface(X, Y, free_energies, 
                             cmap='viridis',
                             linewidth=0,
                             antialiased=True)
        
        ax.set_xlabel('Internal States')
        ax.set_ylabel('External States') 
        ax.set_zlabel('Free Energy')
        ax.set_title(title)
        
        fig.colorbar(surf)
        
        if save_path:
            path = self._ensure_viz_path(save_path)
            plt.savefig(path)
        plt.close()
        
    def plot_belief_updates(self,
                          prior: np.ndarray, 
                          posterior: np.ndarray,
                          observations: np.ndarray,
                          timesteps: np.ndarray,
                          title: str = "Belief Updates Over Time",
                          save_path: Optional[str] = None) -> None:
        """Plot belief distributions before and after updates.
        
        Args:
            prior: Prior belief distributions
            posterior: Posterior belief distributions after updates
            observations: Observed data points
            timesteps: Array of timesteps
            title: Plot title
            save_path: Optional path to save the plot
        """
        logger.info("Plotting belief updates")
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        ax.plot(timesteps, prior, label='Prior', linestyle='--')
        ax.plot(timesteps, posterior, label='Posterior')
        ax.scatter(timesteps, observations, color='red', label='Observations', alpha=0.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Belief Probability')
        ax.set_title(title)
        ax.legend()
        
        if save_path:
            path = self._ensure_viz_path(save_path)
            plt.savefig(path)
        plt.close()
        
    def plot_markov_blanket_states(self,
                                 sensory_states: np.ndarray,
                                 internal_states: np.ndarray, 
                                 active_states: np.ndarray,
                                 title: str = "Markov Blanket State Relationships",
                                 save_path: Optional[str] = None) -> None:
        """Visualize relationships between sensory, internal and active states.
        
        Args:
            sensory_states: Array of sensory state values
            internal_states: Array of internal state values
            active_states: Array of active state values
            title: Plot title 
            save_path: Optional path to save the plot
        """
        logger.info("Plotting Markov blanket states")
        G = nx.Graph()
        
        # Add nodes
        G.add_node("Sensory", pos=(0,0))
        G.add_node("Internal", pos=(1,0)) 
        G.add_node("Active", pos=(0.5,1))
        
        # Add edges
        G.add_edge("Sensory", "Internal")
        G.add_edge("Internal", "Active")
        G.add_edge("Active", "Sensory")
        
        fig, ax = plt.subplots(figsize=self.default_figsize)
        pos = nx.get_node_attributes(G,'pos')
        
        nx.draw(G, pos, 
               with_labels=True,
               node_color='lightblue',
               node_size=3000,
               font_size=10,
               font_weight='bold')
        
        if save_path:
            path = self._ensure_viz_path(save_path)
            plt.savefig(path)
        plt.close()
        
    def plot_message_network_dynamics(self,
                                    messages: List[Dict[str, Any]],
                                    connections: Dict[str, List[str]],
                                    filename: str,
                                    title: str = "Message Network Dynamics") -> None:
        """Visualize message passing dynamics in a network.
        
        Args:
            messages: List of message dictionaries with source, target and content
            connections: Dictionary mapping nodes to their connected nodes
            filename: Name of file to save visualization
            title: Plot title
        """
        logger.info("Plotting message network dynamics")
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges from connections
        for source, targets in connections.items():
            for target in targets:
                G.add_edge(source, target)
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=self.default_figsize)
        pos = nx.spring_layout(G)
        
        # Draw the base network with arrows=True to avoid warning
        nx.draw(G, pos,
               with_labels=True,
               node_color='lightblue',
               node_size=1000,
               arrows=True,  # Fix for NetworkX warning
               arrowsize=20,
               font_size=10,
               font_weight='bold',
               ax=ax)
        
        # Highlight message paths
        edge_colors = []
        edge_weights = []
        
        for edge in G.edges():
            color = 'gray'
            width = 1.0
            
            # Check if this edge appears in messages
            for msg in messages:
                if msg['source'] == edge[0] and msg['target'] == edge[1]:
                    color = 'red'
                    width = 2.0
                    break
                    
            edge_colors.append(color)
            edge_weights.append(width)
        
        # Draw edges with colors indicating message paths
        nx.draw_networkx_edges(G, pos,
                             edge_color=edge_colors,
                             width=edge_weights,
                             arrows=True,  # Fix for NetworkX warning
                             arrowsize=20)
        
        plt.title(title)
        
        # Save the plot using consistent path handling
        path = self._ensure_viz_path(filename)
        plt.savefig(path)
        plt.close() 