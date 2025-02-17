"""
Visualization utilities for the Path Network simulation.
Provides real-time visualization of the network topology and agent states.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Optional, Tuple
from ..core.network import PathNetwork

class NetworkVisualizer:
    """
    Visualizes the network topology and agent states in real-time.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        # Set up style
        try:
            sns.set_style("whitegrid")
            sns.set_context("notebook", font_scale=1.2)
        except Exception as e:
            print(f"Warning: Could not set seaborn style: {e}")
            plt.style.use('default')
        
        self.fig = plt.figure(figsize=figsize)
        
        # Create subplots
        self.network_ax = self.fig.add_subplot(221)
        self.heights_ax = self.fig.add_subplot(222)
        self.history_ax = self.fig.add_subplot(212)
        
        # Initialize data storage
        self.water_level_history: List[float] = []
        self.height_histories: Dict[int, List[float]] = {}
        self.time_points: List[int] = []
        
        # Style settings
        self.fig.tight_layout(pad=3.0)
        
    def update(self, network: PathNetwork, water_level: float) -> None:
        """
        Update the visualization with current network state.
        
        Args:
            network: The current network state
            water_level: Current global water level
        """
        self._clear_axes()
        
        # Get current network state
        graph, heights = network.get_network_state()
        
        # Update histories
        self.water_level_history.append(water_level)
        self.time_points.append(len(self.water_level_history))
        
        for node_id, height in heights.items():
            if node_id not in self.height_histories:
                self.height_histories[node_id] = []
            self.height_histories[node_id].append(height)
        
        # Draw network topology
        self._draw_network(graph, heights)
        
        # Draw current heights
        self._draw_heights(heights, water_level)
        
        # Draw history
        self._draw_history()
        
        # Update display
        try:
            plt.draw()
            plt.pause(0.01)
        except Exception as e:
            print(f"Warning: Could not update display in real-time: {e}")
    
    def _clear_axes(self) -> None:
        """Clear all axes for redrawing."""
        for ax in [self.network_ax, self.heights_ax, self.history_ax]:
            ax.clear()
    
    def _draw_network(self, graph: nx.DiGraph, heights: Dict[int, float]) -> None:
        """Draw the network topology with node colors based on heights."""
        self.network_ax.set_title('Network Topology')
        
        # Calculate node colors based on heights
        vmin = min(heights.values())
        vmax = max(heights.values())
        node_colors = [heights[node] for node in graph.nodes()]
        
        # Calculate edge weights for width
        edge_weights = [
            graph[u][v]['weight'] * 2 for u, v in graph.edges()
        ]
        
        # Draw the network
        pos = nx.spring_layout(graph, seed=42)  # Fixed seed for consistency
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            node_size=500,
            cmap='coolwarm',
            vmin=vmin,
            vmax=vmax,
            ax=self.network_ax
        )
        nx.draw_networkx_edges(
            graph,
            pos,
            edge_color='gray',
            width=edge_weights,
            alpha=0.6,
            ax=self.network_ax
        )
        nx.draw_networkx_labels(
            graph,
            pos,
            {node: str(node) for node in graph.nodes()},
            ax=self.network_ax
        )
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap='coolwarm',
            norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        plt.colorbar(sm, ax=self.network_ax, label='Height')
    
    def _draw_heights(
        self,
        heights: Dict[int, float],
        water_level: float
    ) -> None:
        """Draw current agent heights and water level."""
        self.heights_ax.set_title('Current Heights')
        
        # Plot agent heights
        nodes = list(heights.keys())
        height_values = [heights[node] for node in nodes]
        
        # Create bar plot with custom colors based on height
        colors = plt.cm.coolwarm(
            plt.Normalize()(height_values)
        )
        self.heights_ax.bar(
            nodes,
            height_values,
            alpha=0.6,
            color=colors
        )
        
        # Plot water level
        self.heights_ax.axhline(
            y=water_level,
            color='r',
            linestyle='--',
            label='Water Level'
        )
        
        self.heights_ax.set_xlabel('Agent ID')
        self.heights_ax.set_ylabel('Height')
        self.heights_ax.legend()
        
        # Set y-limits with some padding
        ymin = min(min(height_values), water_level)
        ymax = max(max(height_values), water_level)
        padding = (ymax - ymin) * 0.1
        self.heights_ax.set_ylim(ymin - padding, ymax + padding)
    
    def _draw_history(self) -> None:
        """Draw the history of water level and agent heights."""
        self.history_ax.set_title('Height History')
        
        # Create color palette for agents
        num_agents = len(self.height_histories)
        colors = plt.cm.viridis(np.linspace(0, 1, num_agents))
        
        # Plot agent height histories
        for (node_id, history), color in zip(
            self.height_histories.items(), colors
        ):
            self.history_ax.plot(
                self.time_points[-len(history):],
                history,
                alpha=0.6,
                label=f'Agent {node_id}',
                color=color
            )
        
        # Plot water level history
        self.history_ax.plot(
            self.time_points,
            self.water_level_history,
            'r--',
            label='Water Level',
            linewidth=2,
            alpha=0.8
        )
        
        self.history_ax.set_xlabel('Time Step')
        self.history_ax.set_ylabel('Height')
        
        # Adjust legend
        self.history_ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            ncol=2
        )
    
    def save(self, filename: str) -> None:
        """Save the current figure to a file."""
        try:
            # Adjust layout before saving
            self.fig.tight_layout()
            self.fig.savefig(
                filename,
                bbox_inches='tight',
                dpi=300,
                facecolor='white',
                edgecolor='none'
            )
        except Exception as e:
            print(f"Warning: Could not save figure to {filename}: {e}")
    
    def close(self) -> None:
        """Close the figure."""
        plt.close(self.fig) 