"""
Nestmate agent implementation using active inference.
"""

from enum import Enum
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from ant_colony.environment.world import Position, Resource

class TaskType(Enum):
    """Types of tasks an ant can perform."""
    FORAGING = 'foraging'
    MAINTENANCE = 'maintenance'
    NURSING = 'nursing'
    DEFENSE = 'defense'
    EXPLORATION = 'exploration'

@dataclass
class Belief:
    """Represents an agent's beliefs about the world state."""
    food_location: Optional[Position] = None
    nest_location: Optional[Position] = None
    danger_level: float = 0.0
    energy_level: float = 1.0
    task_urgency: Dict[TaskType, float] = field(default_factory=lambda: {task: 0.0 for task in TaskType})

class Nestmate:
    """
    Implementation of an ant agent using active inference principles.
    """
    
    def __init__(self, config: dict):
        """Initialize the agent."""
        self.config = config
        
        # Initialize beliefs first
        self.beliefs = Belief()
        
        # Physical state
        self.position = None
        self.orientation = 0.0
        self.speed = 0.0
        self.energy = config['physical']['energy']['initial']
        
        # Carrying state
        self.carrying: Optional[Resource] = None
        
        # Task state
        self.current_task = TaskType.EXPLORATION
        self.task_time = 0.0
        
        # Sensory state
        self.observations = {
            'pheromones': {},
            'resources': [],
            'nestmates': [],
            'terrain': None
        }
        
        # Internal model
        self.preferences = {task: 1.0 for task in TaskType}
        
        # Learning parameters
        self.learning_rate = config['behavior'].get('learning_rate', 0.1)
        self.exploration_rate = config['behavior'].get('exploration_rate', 0.2)
        
        # ... existing code ... 