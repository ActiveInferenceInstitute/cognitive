"""
Visualization module for the ant colony simulation using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation

class SimulationRenderer:
    """Handles visualization of the ant colony simulation."""
    
    def __init__(self, config: dict):
        """Initialize the renderer with configuration settings."""
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.env_size = config['environment']['size']
        
        # Default colors if not specified in config
        self.colors = {
            'nest': 'brown',
            'food': 'green',
            'obstacles': 'gray',
            'agents': {
                'foraging': 'red',
                'maintenance': 'blue',
                'nursing': 'pink',
                'defense': 'darkred',
                'exploration': 'orange'
            },
            'pheromones': {
                'food': 'green',
                'home': 'red',
                'alarm': 'orange',
                'trail': 'blue'
            }
        }
        
        # Override defaults with config values if present
        if 'colors' in self.viz_config:
            self.colors.update(self.viz_config['colors'])
        
        # Setup the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(0, self.env_size[0])
        self.ax.set_ylim(0, self.env_size[1])
        self.ax.set_aspect('equal')
        
        # Initialize collections for different elements
        self.agent_patches = []
        self.food_patches = []
        self.obstacle_patches = []
        self.pheromone_plots = {}
        
        # Setup the nest
        nest_loc = config['environment']['nest_location']
        nest = Circle(nest_loc, 5, color=self.colors['nest'])
        self.ax.add_patch(nest)
        
    def update(self, world_state: dict) -> None:
        """Update the visualization with current world state."""
        # Clear previous patches
        for patch in self.agent_patches + self.food_patches + self.obstacle_patches:
            patch.remove()
        self.agent_patches.clear()
        self.food_patches.clear()
        self.obstacle_patches.clear()
        
        # Update pheromones
        for p_type, grid in world_state['pheromones'].items():
            if p_type not in self.pheromone_plots:
                color = self.colors['pheromones'][p_type]
                self.pheromone_plots[p_type] = self.ax.imshow(
                    grid,
                    extent=[0, self.env_size[0], 0, self.env_size[1]],
                    cmap='Greens' if p_type == 'food' else 'Reds',
                    alpha=0.3,
                    vmin=0,
                    vmax=1
                )
            else:
                self.pheromone_plots[p_type].set_array(grid)
        
        # Draw agents
        for agent in world_state['agents']:
            color = self.colors['agents'][agent.current_task.value]
            agent_patch = self._create_agent_patch(agent, color)
            self.ax.add_patch(agent_patch)
            self.agent_patches.append(agent_patch)
        
        # Draw food sources
        for food in world_state['resources']:
            food_patch = Circle(
                (food.position.x, food.position.y),
                food.size if hasattr(food, 'size') else 1.0,
                color=self.colors['food']
            )
            self.ax.add_patch(food_patch)
            self.food_patches.append(food_patch)
        
        # Draw obstacles
        for obstacle in world_state['obstacles']:
            obstacle_patch = Circle(
                (obstacle.position.x, obstacle.position.y),
                obstacle.size if hasattr(obstacle, 'size') else 1.0,
                color=self.colors['obstacles']
            )
            self.ax.add_patch(obstacle_patch)
            self.obstacle_patches.append(obstacle_patch)
        
        # Trigger redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def _create_agent_patch(self, agent, color: str) -> Wedge:
        """Create a wedge patch to represent an agent."""
        # Create a wedge shape to show orientation
        radius = 1.0
        angle = np.degrees(agent.position.theta)
        wedge = Wedge(
            (agent.position.x, agent.position.y),
            radius,
            angle - 30,  # 60 degree wide wedge
            angle + 30,
            color=color
        )
        return wedge
        
    def save_animation(self, frames: list, filename: str) -> None:
        """Save the simulation as an animation."""
        anim = animation.ArtistAnimation(
            self.fig,
            frames,
            interval=50,
            blit=True,
            repeat=False
        )
        anim.save(filename, writer='pillow')
        
    def show(self) -> None:
        """Display the current visualization."""
        plt.show()
        
    def close(self) -> None:
        """Close the visualization window."""
        plt.close(self.fig) 