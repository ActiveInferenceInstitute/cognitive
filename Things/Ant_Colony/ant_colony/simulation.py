"""
Main simulation coordinator for ant colony simulation.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

from ant_colony.environment import World
from ant_colony.colony import Colony
from ant_colony.visualization import ColonyVisualizer
from ant_colony.utils.data_collection import DataCollector

class Simulation:
    """Main simulation coordinator."""
    
    def __init__(self, config_path: str):
        """Initialize simulation with configuration."""
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set random seed
        np.random.seed(self.config['simulation']['random_seed'])
        
        # Initialize components
        self.environment = World(self.config['environment'])
        self.colony = Colony(self.config['colony'], self.environment)
        
        # Initialize visualization if enabled
        if self.config.get('visualization', {}).get('enabled', True):
            self.visualizer = ColonyVisualizer(self.config)
        else:
            self.visualizer = None
            
        # Initialize data collection if enabled
        if self.config.get('data_collection', {}).get('enabled', True):
            self.data_collector = DataCollector(self.config['data_collection'])
        else:
            self.data_collector = None
        
        # Simulation state
        self.current_step = 0
        self.current_time = 0.0
        self.paused = False
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def step(self) -> None:
        """Execute one simulation step."""
        if self.paused:
            return
            
        # Get timestep
        dt = self.config['simulation']['timestep']
        
        # Update environment
        self.environment.update(dt)
        
        # Update colony
        self.colony.update(dt)
        
        # Update visualization if enabled
        if self.visualizer is not None:
            world_state = {
                'agents': self.colony.agents,
                'resources': self.environment.resources,
                'obstacles': [],  # TODO: Add obstacles
                'pheromones': self.environment.pheromones
            }
            self.visualizer.update(world_state)
            
        # Collect data if enabled
        if self.data_collector is not None:
            self.data_collector.collect(self.current_time, world_state)
            
            # Save data periodically
            if self.current_step % self.config['data_collection']['save_interval'] == 0:
                self.data_collector.save_data()
                
        # Update time
        self.current_time += dt
        self.current_step += 1
        
    def run(self, headless: bool = False) -> None:
        """Run the simulation."""
        max_steps = self.config['simulation']['max_steps']
        
        try:
            while self.current_step < max_steps and not self.paused:
                self.step()
                
                # Print progress
                if self.current_step % 100 == 0:
                    print(f"Step {self.current_step}/{max_steps}")
                    
                # Update visualization
                if not headless and self.visualizer is not None:
                    plt.pause(0.001)  # Allow GUI to update
                    
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
            
        finally:
            # Save final data
            if self.data_collector is not None:
                self.data_collector.save_data()
                
            # Show final visualization
            if not headless and self.visualizer is not None:
                plt.show()
                
    def pause(self) -> None:
        """Pause the simulation."""
        self.paused = True
        
    def resume(self) -> None:
        """Resume the simulation."""
        self.paused = False
        
    def reset(self) -> None:
        """Reset simulation to initial state."""
        # Reset time
        self.current_step = 0
        self.current_time = 0.0
        
        # Reset components
        self.environment = World(self.config['environment'])
        self.colony = Colony(self.config['colony'], self.environment)
        
        # Clear data
        if self.data_collector is not None:
            self.data_collector = DataCollector(self.config['data_collection']) 