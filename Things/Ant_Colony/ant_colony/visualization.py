"""
Enhanced Visualization Module for Ant Colony Simulation

This module provides advanced visualization capabilities for the ant colony simulation,
including animated pheromone trails, agent movements, and colony statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, List, Optional
from matplotlib.patches import Circle, Wedge, Path, PathPatch
from matplotlib.collections import PatchCollection
import colorsys
import networkx as nx

class ColonyVisualizer:
    """Advanced visualization for the ant colony simulation."""
    
    def __init__(self, config: dict):
        """Initialize visualizer."""
        self.config = config
        self.fig = None
        self.axes = {}
        self.artists = {}
        self.animation = None
        
        # Color schemes
        self.color_schemes = {
            'terrain': plt.cm.terrain,
            'pheromone': {
                'food': plt.cm.Greens,
                'home': plt.cm.Reds,
                'alarm': plt.cm.Oranges,
                'trail': plt.cm.Blues
            },
            'agents': {
                'foraging': '#2ecc71',
                'maintenance': '#3498db',
                'nursing': '#9b59b6',
                'defense': '#e74c3c',
                'exploration': '#f1c40f'
            }
        }
        
        # Setup visualization
        self._setup_visualization()
        
    def _setup_visualization(self):
        """Set up the visualization layout."""
        plt.style.use('dark_background')
        
        # Create figure with custom layout
        self.fig = plt.figure(figsize=(16, 9))
        gs = self.fig.add_gridspec(3, 3)
        
        # Main simulation view (larger)
        self.axes['main'] = self.fig.add_subplot(gs[0:2, 0:2])
        self.axes['main'].set_title('Colony Simulation')
        
        # Pheromone levels
        self.axes['pheromone'] = self.fig.add_subplot(gs[0, 2])
        self.axes['pheromone'].set_title('Pheromone Levels')
        
        # Resource levels
        self.axes['resources'] = self.fig.add_subplot(gs[1, 2])
        self.axes['resources'].set_title('Resource Levels')
        
        # Task distribution
        self.axes['tasks'] = self.fig.add_subplot(gs[2, 0])
        self.axes['tasks'].set_title('Task Distribution')
        
        # Efficiency metrics
        self.axes['metrics'] = self.fig.add_subplot(gs[2, 1])
        self.axes['metrics'].set_title('Colony Metrics')
        
        # Network view
        self.axes['network'] = self.fig.add_subplot(gs[2, 2])
        self.axes['network'].set_title('Social Network')
        
        plt.tight_layout()
        
    def create_animation(self, simulation, interval: int = 50) -> FuncAnimation:
        """Create animation of the colony simulation."""
        def update(frame):
            # Update simulation state
            if not simulation.paused:
                simulation.step()
            
            # Clear all axes
            for ax in self.axes.values():
                ax.clear()
                
            # Update all plots
            artists = []
            artists.extend(self._plot_main_view(simulation))
            artists.extend(self._plot_pheromone_levels(simulation))
            artists.extend(self._plot_resource_levels(simulation))
            artists.extend(self._plot_task_distribution(simulation))
            artists.extend(self._plot_efficiency_metrics(simulation))
            artists.extend(self._plot_social_network(simulation))
            
            return artists
            
        self.animation = FuncAnimation(
            self.fig,
            update,
            frames=None,
            interval=interval,
            blit=True
        )
        
        return self.animation
        
    def _plot_main_view(self, simulation) -> List:
        """Plot main simulation view with enhanced graphics."""
        ax = self.axes['main']
        artists = []
        
        # Plot terrain with enhanced colormap
        terrain = simulation.environment.terrain.height_map
        terrain_img = ax.imshow(terrain, cmap=self.color_schemes['terrain'])
        artists.append(terrain_img)
        
        # Plot pheromone trails with alpha blending
        for ptype, pdata in simulation.environment.pheromones.layers.items():
            pheromone_grid = pdata['grid']
            if np.any(pheromone_grid > 0):
                pheromone_img = ax.imshow(
                    pheromone_grid,
                    cmap=self.color_schemes['pheromone'][ptype],
                    alpha=0.5
                )
                artists.append(pheromone_img)
        
        # Plot agents with directional markers
        for agent in simulation.colony.agents:
            agent_color = self.color_schemes['agents'][agent.current_task.value]
            
            # Create ant shape
            ant = self._create_ant_shape(agent.position, agent.orientation, 
                                       size=1.0, color=agent_color)
            ax.add_patch(ant)
            artists.append(ant)
            
            # Add carrying indicator if needed
            if agent.carrying is not None:
                carry_indicator = Circle(
                    (agent.position[0], agent.position[1]),
                    radius=0.3,
                    color='yellow',
                    alpha=0.7
                )
                ax.add_patch(carry_indicator)
                artists.append(carry_indicator)
        
        # Plot resources
        for resource in simulation.environment.resources:
            if resource.type == 'food':
                marker = '*'
                color = 'yellow'
            else:
                marker = 's'
                color = 'cyan'
                
            resource_dot = ax.scatter(
                resource.position.x,
                resource.position.y,
                c=color,
                marker=marker,
                s=50
            )
            artists.append(resource_dot)
        
        # Plot nest with concentric circles
        nest_x = simulation.colony.nest_position.x
        nest_y = simulation.colony.nest_position.y
        
        for radius in [1, 2, 3]:
            nest_circle = Circle(
                (nest_x, nest_y),
                radius,
                color='white',
                fill=False,
                alpha=0.5
            )
            ax.add_patch(nest_circle)
            artists.append(nest_circle)
            
        # Add nest center
        nest_center = ax.scatter(
            [nest_x], [nest_y],
            c='white',
            marker='s',
            s=100
        )
        artists.append(nest_center)
        
        # Set limits and title
        ax.set_xlim(0, simulation.environment.size[0])
        ax.set_ylim(0, simulation.environment.size[1])
        ax.set_title(f'Colony Simulation (Step {simulation.current_step})')
        
        return artists
        
    def _create_ant_shape(self, position, orientation, size=1.0, color='white'):
        """Create an ant-like shape for visualization."""
        # Create ant body segments
        body_verts = [
            (-0.5, -0.2),  # Abdomen
            (0.0, -0.1),   # Thorax
            (0.5, 0.0),    # Head
            (0.0, 0.1),
            (-0.5, 0.2)
        ]
        
        # Scale and rotate vertices
        cos_theta = np.cos(orientation)
        sin_theta = np.sin(orientation)
        
        transformed_verts = []
        for x, y in body_verts:
            x_rot = x * cos_theta - y * sin_theta
            y_rot = x * sin_theta + y * cos_theta
            transformed_verts.append(
                (position[0] + x_rot * size, 
                 position[1] + y_rot * size)
            )
            
        # Create path
        codes = [Path.MOVETO] + [Path.LINETO] * (len(transformed_verts) - 1)
        path = Path(transformed_verts, codes)
        
        return PathPatch(path, facecolor=color, edgecolor='none', alpha=0.7)
        
    def _plot_pheromone_levels(self, simulation) -> List:
        """Plot pheromone concentration levels."""
        ax = self.axes['pheromone']
        artists = []
        
        # Get pheromone data
        pheromone_levels = []
        labels = []
        colors = []
        
        for ptype, pdata in simulation.environment.pheromones.layers.items():
            pheromone_levels.append(np.mean(pdata['grid']))
            labels.append(ptype)
            colors.append(self.color_schemes['agents'].get(ptype, '#95a5a6'))
            
        # Create bar plot
        if pheromone_levels:
            bars = ax.bar(range(len(pheromone_levels)), pheromone_levels,
                         color=colors)
            artists.extend(bars)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45)
            ax.set_ylabel('Average Concentration')
            
        return artists
        
    def _plot_resource_levels(self, simulation) -> List:
        """Plot resource levels over time."""
        ax = self.axes['resources']
        artists = []
        
        if len(simulation.data['time']) > 0:
            for resource_type in simulation.colony.resources.keys():
                values = [d[resource_type] for d in simulation.data['resources']]
                line = ax.plot(simulation.data['time'][-100:], values[-100:],
                             label=resource_type)
                artists.extend(line)
                
            ax.legend()
            ax.set_xlabel('Time')
            ax.set_ylabel('Amount')
            
        return artists
        
    def _plot_task_distribution(self, simulation) -> List:
        """Plot task distribution with enhanced graphics."""
        ax = self.axes['tasks']
        artists = []
        
        if len(simulation.data['task_distribution']) > 0:
            latest_dist = simulation.data['task_distribution'][-1]
            tasks = list(latest_dist.keys())
            counts = [latest_dist[task] for task in tasks]
            colors = [self.color_schemes['agents'][task.value] for task in tasks]
            
            # Create stacked bars for current and needed agents
            bars = ax.bar(range(len(tasks)), counts, color=colors)
            artists.extend(bars)
            
            # Add target lines for needed agents
            for i, task in enumerate(tasks):
                need = simulation.colony.task_needs[task] * len(simulation.colony.agents)
                line = ax.hlines(need, i-0.4, i+0.4, colors='white', linestyles='--')
                artists.append(line)
                
            ax.set_xticks(range(len(tasks)))
            ax.set_xticklabels([task.value for task in tasks], rotation=45)
            
        return artists
        
    def _plot_efficiency_metrics(self, simulation) -> List:
        """Plot efficiency metrics with enhanced graphics."""
        ax = self.axes['metrics']
        artists = []
        
        if len(simulation.data['efficiency_metrics']) > 0:
            metrics = simulation.data['efficiency_metrics'][-1]
            values = list(metrics.values())
            labels = list(metrics.keys())
            
            # Create radar chart
            angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
            values = np.concatenate((values, [values[0]]))  # Close the polygon
            angles = np.concatenate((angles, [angles[0]]))  # Close the polygon
            
            # Plot radar
            line = ax.plot(angles, values)
            artists.extend(line)
            
            # Fill radar
            fill = ax.fill(angles, values, alpha=0.25)
            artists.extend(fill)
            
            # Add labels
            for angle, label in zip(angles[:-1], labels):
                ha = 'right' if np.cos(angle) < 0 else 'left'
                va = 'top' if np.sin(angle) < 0 else 'bottom'
                
                ax.text(angle, 1.2, label,
                       ha=ha, va=va,
                       rotation=np.degrees(angle))
                
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            
        return artists
        
    def _plot_social_network(self, simulation) -> List:
        """Plot social network with enhanced graphics."""
        ax = self.axes['network']
        artists = []
        
        # Get network data
        G = simulation.colony.interaction_network
        
        if len(G.nodes) > 0:
            # Calculate node positions using spring layout
            pos = nx.spring_layout(G)
            
            # Draw edges
            for edge in G.edges():
                x = [pos[edge[0]][0], pos[edge[1]][0]]
                y = [pos[edge[0]][1], pos[edge[1]][1]]
                line = ax.plot(x, y, 'w-', alpha=0.2)
                artists.extend(line)
            
            # Draw nodes
            for node in G.nodes():
                color = self.color_schemes['agents'][node.current_task.value]
                dot = ax.scatter(pos[node][0], pos[node][1],
                               c=color, s=50)
                artists.append(dot)
                
        ax.set_xticks([])
        ax.set_yticks([])
        
        return artists 