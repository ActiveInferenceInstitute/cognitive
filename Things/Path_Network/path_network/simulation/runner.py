"""
Main simulation runner for the Path Network.
Orchestrates the interaction between the network of agents and environmental dynamics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ..core.network import PathNetwork, NetworkConfig
from ..core.dynamics import EnvironmentalDynamics, DynamicsConfig
from ..utils.visualization import NetworkVisualizer

@dataclass
class SimulationConfig:
    """Configuration for the simulation."""
    network_config: NetworkConfig
    dynamics_config: DynamicsConfig
    num_steps: int = 1000
    save_interval: int = 10
    visualization_interval: int = 50

class SimulationRunner:
    """
    Main simulation runner that coordinates the network of agents
    and environmental dynamics.
    """
    
    def __init__(
        self,
        config: SimulationConfig,
        visualizer: Optional[NetworkVisualizer] = None
    ):
        self.config = config
        self.network = PathNetwork(config.network_config)
        self.dynamics = EnvironmentalDynamics(config.dynamics_config)
        self.visualizer = visualizer
        
        self.current_step = 0
        self.history: List[Dict] = []
    
    def step(self) -> Dict:
        """
        Perform one step of the simulation.
        
        Returns:
            Dict containing the current state of the simulation
        """
        # Get current water level
        water_level = self.dynamics.step()
        
        # Update network
        agent_heights = self.network.step(water_level)
        
        # Record state
        state = {
            'step': self.current_step,
            'water_level': water_level,
            'agent_heights': agent_heights
        }
        
        if self.current_step % self.config.save_interval == 0:
            self.history.append(state)
        
        # Visualize if needed
        if (
            self.visualizer is not None and
            self.current_step % self.config.visualization_interval == 0
        ):
            self.visualizer.update(self.network, water_level)
        
        self.current_step += 1
        return state
    
    def run(self, num_steps: Optional[int] = None) -> List[Dict]:
        """
        Run the simulation for a specified number of steps.
        
        Args:
            num_steps: Number of steps to run (default: config.num_steps)
            
        Returns:
            List of recorded states
        """
        steps_to_run = num_steps or self.config.num_steps
        
        for _ in range(steps_to_run):
            self.step()
        
        return self.history
    
    def add_perturbation(
        self,
        magnitude: float,
        duration: float,
        decay: float = 1.0
    ) -> None:
        """Add a perturbation to the environmental dynamics."""
        self.dynamics.add_perturbation(magnitude, duration, decay)
    
    def get_network_state(self) -> Tuple[Dict[int, float], float]:
        """
        Get the current state of the network and environment.
        
        Returns:
            Tuple of agent heights and current water level
        """
        _, heights = self.network.get_network_state()
        water_level = self.dynamics.history[-1] if self.dynamics.history else 0.0
        return heights, water_level
    
    def get_history(self) -> List[Dict]:
        """Get the simulation history."""
        return self.history.copy()
    
    def reset(self) -> None:
        """Reset the simulation to initial state."""
        self.current_step = 0
        self.history.clear()
        self.dynamics.reset()
        # Note: Network is not reset as it maintains its learned structure 