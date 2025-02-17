"""
Active Inference Agent implementation for the Path Network simulation.
This module contains the core agent class that performs continuous-time active inference.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class AgentConfig:
    """Configuration parameters for an Active Inference agent."""
    dt: float = 0.01  # Integration time step
    taylor_order: int = 2  # Order of Taylor expansion for generalized coordinates
    tolerance: float = 0.1  # Tolerance range for water level
    learning_rate: float = 0.01  # Learning rate for parameter updates
    initial_height: Optional[float] = None  # Initial height of the agent

class ActiveInferenceAgent:
    """
    An Active Inference agent that infers and responds to water level changes.
    
    The agent uses generalized coordinates and continuous-time inference to
    maintain its height within a tolerable range of the global water level.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.height = (
            config.initial_height 
            if config.initial_height is not None 
            else np.random.normal(0, 0.1)
        )
        
        # Generalized coordinates for position (height)
        self.gen_coords = torch.zeros(config.taylor_order)
        self.gen_coords[0] = self.height
        
        # Free energy components
        self.mu = torch.zeros(config.taylor_order)  # Expected states
        self.pi = torch.ones(config.taylor_order)   # Precision (inverse variance)
        
        # History for analysis
        self.height_history: List[float] = [self.height]
        self.prediction_error_history: List[float] = []
        
    def update_beliefs(self, sensory_input: float) -> None:
        """
        Update beliefs about the environment based on sensory input.
        
        Args:
            sensory_input: The current water level measurement
        """
        # Compute prediction error
        prediction_error = sensory_input - self.mu[0]
        self.prediction_error_history.append(prediction_error.item())
        
        # Update generalized coordinates
        for i in range(self.config.taylor_order - 1):
            self.gen_coords[i] += (
                self.gen_coords[i + 1] * self.config.dt +
                self.config.learning_rate * prediction_error * self.pi[i]
            )
        
        # Update expectations
        self.mu = self.gen_coords.clone()
        
    def act(self, water_level: float) -> float:
        """
        Generate an action (height adjustment) based on current beliefs.
        
        Args:
            water_level: Current global water level
            
        Returns:
            float: The new height of the agent
        """
        # Update beliefs based on sensory input
        self.update_beliefs(water_level)
        
        # Compute desired height adjustment
        error = water_level - self.height
        adjustment = np.clip(
            error * self.config.learning_rate,
            -self.config.tolerance,
            self.config.tolerance
        )
        
        # Update height
        self.height += adjustment
        self.height_history.append(self.height)
        
        return self.height
    
    def get_state(self) -> Tuple[float, List[float]]:
        """
        Get the current state of the agent.
        
        Returns:
            Tuple containing current height and generalized coordinates
        """
        return self.height, self.gen_coords.tolist()
    
    def get_history(self) -> Tuple[List[float], List[float]]:
        """
        Get the agent's history of heights and prediction errors.
        
        Returns:
            Tuple containing height history and prediction error history
        """
        return self.height_history, self.prediction_error_history 