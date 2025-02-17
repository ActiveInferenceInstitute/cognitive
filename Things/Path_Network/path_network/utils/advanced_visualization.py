"""
Advanced visualization utilities for the Path Network simulation.
Provides comprehensive visualization options including animations and 3D plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.stats import gaussian_kde
from typing import Dict, List, Optional, Tuple, Any
import yaml
from pathlib import Path
from tqdm import tqdm
import imageio
from sklearn.decomposition import PCA
from ..core.network import PathNetwork

class AdvancedVisualizer:
    """Advanced visualization tools for the Path Network simulation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['visualization']
        
        # Set style
        plt.style.use(self.config['style'])
        sns.set_context("notebook", font_scale=self.config['font_scale'])
        
        # Initialize storage for animation frames
        self.network_frames: List[plt.Figure] = []
        self.phase_space_frames: List[plt.Figure] = []
    
    def create_network_animation(
        self,
        history: List[Dict],
        output_path: Path
    ) -> None:
        """Create an animated visualization of the network evolution."""
        fig = plt.figure(figsize=self.config['animation_figsize'])
        
        def update(frame):
            plt.clf()
            state = history[frame]
            self._draw_network_state(state, fig.gca())
            return fig.gca()
        
        anim = FuncAnimation(
            fig,
            update,
            frames=len(history),
            interval=1000/self.config['fps']
        )
        
        # Save as GIF
        writer = PillowWriter(fps=self.config['fps'])
        anim.save(output_path / 'network_evolution.gif', writer=writer)
        plt.close()
    
    def create_phase_space_animation(
        self,
        history: List[Dict],
        output_path: Path
    ) -> None:
        """Create an animated 3D phase space visualization."""
        if not self.config['enable_3d']:
            return
            
        fig = plt.figure(figsize=self.config['animation_figsize'])
        ax = fig.add_subplot(111, projection='3d')
        
        # Compute PCA for dimensionality reduction
        heights_matrix = []
        for state in history:
            heights = list(state['agent_heights'].values())
            heights_matrix.append(heights)
        
        pca = PCA(n_components=3)
        transformed = pca.fit_transform(heights_matrix)
        
        def update(frame):
            ax.clear()
            ax.scatter(
                transformed[:frame+1, 0],
                transformed[:frame+1, 1],
                transformed[:frame+1, 2],
                c=range(frame+1),
                cmap='viridis'
            )
            ax.view_init(
                elev=30,
                azim=frame * self.config['3d_rotation_speed']
            )
            return ax
        
        anim = FuncAnimation(
            fig,
            update,
            frames=len(history),
            interval=1000/self.config['fps']
        )
        
        anim.save(output_path / 'phase_space.gif', writer=PillowWriter(fps=self.config['fps']))
        plt.close()
    
    def create_spectral_analysis(
        self,
        history: List[Dict],
        output_path: Path
    ) -> None:
        """Create spectral analysis visualizations."""
        water_levels = [state['water_level'] for state in history]
        
        # Compute spectrograms for water level
        f, t, Sxx = signal.spectrogram(water_levels, fs=1.0)
        
        plt.figure(figsize=self.config['timeseries_figsize'])
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency')
        plt.xlabel('Time')
        plt.title('Water Level Spectrogram')
        plt.colorbar(label='Intensity')
        plt.savefig(output_path / 'water_level_spectrogram.png', dpi=self.config['dpi'])
        plt.close()
        
        # Compute and plot power spectrum
        plt.figure(figsize=self.config['timeseries_figsize'])
        f, Pxx = signal.welch(water_levels)
        plt.semilogy(f, Pxx)
        plt.xlabel('Frequency')
        plt.ylabel('Power Spectral Density')
        plt.title('Water Level Power Spectrum')
        plt.savefig(output_path / 'power_spectrum.png', dpi=self.config['dpi'])
        plt.close()
    
    def create_correlation_analysis(
        self,
        history: List[Dict],
        output_path: Path
    ) -> None:
        """Create correlation analysis visualizations."""
        # Extract agent height histories
        agent_histories = {}
        for state in history:
            for agent_id, height in state['agent_heights'].items():
                if agent_id not in agent_histories:
                    agent_histories[agent_id] = []
                agent_histories[agent_id].append(height)
        
        # Compute correlation matrix
        num_agents = len(agent_histories)
        corr_matrix = np.zeros((num_agents, num_agents))
        
        for i in range(num_agents):
            for j in range(num_agents):
                corr = np.corrcoef(
                    agent_histories[i],
                    agent_histories[j]
                )[0, 1]
                corr_matrix[i, j] = corr
        
        # Plot correlation matrix
        plt.figure(figsize=self.config['network_figsize'])
        sns.heatmap(
            corr_matrix,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt='.2f'
        )
        plt.title('Agent Height Correlation Matrix')
        plt.savefig(output_path / 'correlation_matrix.png', dpi=self.config['dpi'])
        plt.close()
    
    def create_interactive_dashboard(
        self,
        history: List[Dict],
        output_path: Path
    ) -> None:
        """Create an interactive HTML dashboard using plotly."""
        if not self.config['enable_interactive']:
            return
            
        # Create subplot figure
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'heatmap'}],
                  [{'type': 'scatter'}, {'type': 'histogram'}]],
            subplot_titles=(
                '3D Phase Space',
                'Agent Correlation',
                'Height Evolution',
                'Height Distribution'
            )
        )
        
        # Add traces
        self._add_phase_space_trace(fig, history, row=1, col=1)
        self._add_correlation_trace(fig, history, row=1, col=2)
        self._add_height_evolution_trace(fig, history, row=2, col=1)
        self._add_height_distribution_trace(fig, history, row=2, col=2)
        
        # Update layout
        fig.update_layout(height=1000, showlegend=True)
        
        # Save
        fig.write_html(output_path / 'interactive_dashboard.html')
    
    def _add_phase_space_trace(
        self,
        fig: go.Figure,
        history: List[Dict],
        row: int,
        col: int
    ) -> None:
        """Add 3D phase space trace to plotly figure."""
        heights_matrix = []
        for state in history:
            heights = list(state['agent_heights'].values())
            heights_matrix.append(heights)
        
        pca = PCA(n_components=3)
        transformed = pca.fit_transform(heights_matrix)
        
        fig.add_trace(
            go.Scatter3d(
                x=transformed[:, 0],
                y=transformed[:, 1],
                z=transformed[:, 2],
                mode='lines+markers',
                marker=dict(
                    size=2,
                    color=range(len(transformed)),
                    colorscale='Viridis',
                    opacity=0.8
                ),
                name='Phase Space'
            ),
            row=row,
            col=col
        )
    
    def _add_correlation_trace(
        self,
        fig: go.Figure,
        history: List[Dict],
        row: int,
        col: int
    ) -> None:
        """Add correlation matrix trace to plotly figure."""
        agent_histories = {}
        for state in history:
            for agent_id, height in state['agent_heights'].items():
                if agent_id not in agent_histories:
                    agent_histories[agent_id] = []
                agent_histories[agent_id].append(height)
        
        num_agents = len(agent_histories)
        corr_matrix = np.zeros((num_agents, num_agents))
        
        for i in range(num_agents):
            for j in range(num_agents):
                corr = np.corrcoef(
                    agent_histories[i],
                    agent_histories[j]
                )[0, 1]
                corr_matrix[i, j] = corr
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix,
                colorscale='RdBu',
                zmid=0,
                name='Correlation'
            ),
            row=row,
            col=col
        )
    
    def _add_height_evolution_trace(
        self,
        fig: go.Figure,
        history: List[Dict],
        row: int,
        col: int
    ) -> None:
        """Add height evolution traces to plotly figure."""
        water_levels = [state['water_level'] for state in history]
        time_points = list(range(len(history)))
        
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=water_levels,
                mode='lines',
                name='Water Level',
                line=dict(color='red', dash='dash')
            ),
            row=row,
            col=col
        )
        
        for agent_id in history[0]['agent_heights'].keys():
            heights = [
                state['agent_heights'][agent_id]
                for state in history
            ]
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=heights,
                    mode='lines',
                    name=f'Agent {agent_id}',
                    opacity=0.6
                ),
                row=row,
                col=col
            )
    
    def _add_height_distribution_trace(
        self,
        fig: go.Figure,
        history: List[Dict],
        row: int,
        col: int
    ) -> None:
        """Add height distribution trace to plotly figure."""
        all_heights = []
        for state in history:
            all_heights.extend(state['agent_heights'].values())
        
        fig.add_trace(
            go.Histogram(
                x=all_heights,
                nbinsx=30,
                name='Height Distribution'
            ),
            row=row,
            col=col
        )
    
    def _draw_network_state(
        self,
        state: Dict,
        ax: plt.Axes
    ) -> None:
        """Draw network state for animation frame."""
        graph = state['network']
        heights = state['agent_heights']
        
        pos = nx.spring_layout(graph, seed=42)
        
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=[heights[node] for node in graph.nodes()],
            node_size=500,
            cmap=self.config['network_cmap'],
            ax=ax
        )
        
        nx.draw_networkx_edges(
            graph,
            pos,
            edge_color='gray',
            width=[
                graph[u][v]['weight'] * 2
                for u, v in graph.edges()
            ],
            alpha=0.6,
            ax=ax
        )
        
        nx.draw_networkx_labels(
            graph,
            pos,
            {node: str(node) for node in graph.nodes()},
            ax=ax
        ) 