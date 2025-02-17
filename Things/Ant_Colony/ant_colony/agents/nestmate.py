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
        
        # Memory state
        self.memory = {
            'temporal': [],  # List of temporal memories (past states/actions)
            'spatial': {},   # Dict of spatial memories (locations)
            'social': {}     # Dict of social memories (interactions)
        }
        
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
        
    def update(self, dt: float, world_state: dict) -> None:
        """Update agent state and take actions."""
        # Update physical state
        self._update_physical_state(dt)
        
        # Update observations
        self._update_observations(world_state)
        
        # Update beliefs through active inference
        self._update_beliefs()
        
        # Select and execute actions
        self._select_actions()
        
        # Update memory with current state
        self._update_memory()
        
    def _update_memory(self) -> None:
        """Update agent's memory with current state."""
        # Add current state to temporal memory
        current_state = {
            'state': self.current_task,
            'position': self.position,
            'energy': self.energy,
            'time': self.task_time
        }
        self.memory['temporal'].append(current_state)
        
        # Limit memory size to prevent unbounded growth
        max_memory = 100
        if len(self.memory['temporal']) > max_memory:
            self.memory['temporal'] = self.memory['temporal'][-max_memory:]
            
        # Update spatial memory if we found something interesting
        if self.beliefs.food_location:
            self.memory['spatial']['food'] = self.beliefs.food_location
            
        # Could add more memory updates here (social interactions, etc.)
        
    def _update_physical_state(self, dt: float) -> None:
        """Update physical state of the agent."""
        # Update position based on velocity and orientation
        if self.speed > 0 and self.position is not None:
            dx = self.speed * np.cos(self.orientation) * dt
            dy = self.speed * np.sin(self.orientation) * dt
            
            self.position.x += dx
            self.position.y += dy
            
        # Update energy
        consumption_rate = self.config['physical']['energy']['consumption_rate']
        self.energy -= consumption_rate * dt
        
        # Check energy level
        if self.energy <= self.config['physical']['energy']['critical_level']:
            self.current_task = TaskType.FORAGING
            
    def _update_observations(self, world_state: dict) -> None:
        """Update agent's observations of the environment."""
        sensor_range = self.config['physical']['sensor_range']
        if hasattr(world_state, 'get_local_state'):
            self.observations = world_state.get_local_state(self.position, sensor_range)
        else:
            # Simplified observation update if get_local_state is not available
            self.observations = {
                'pheromones': world_state.get('pheromones', {}),
                'resources': [r for r in world_state.get('resources', [])
                            if self._distance_to(r.position) <= sensor_range],
                'nestmates': [a for a in world_state.get('agents', [])
                            if a != self and self._distance_to(a.position) <= sensor_range],
                'terrain': None
            }
        
    def _update_beliefs(self) -> None:
        """Update beliefs using active inference."""
        # Update food location belief
        if self.observations['resources']:
            nearest_food = min(
                [r for r in self.observations['resources'] if r.type == 'food'],
                key=lambda r: self._distance_to(r.position),
                default=None
            )
            if nearest_food:
                self.beliefs.food_location = nearest_food.position
                
        # Update danger level belief
        danger_signals = sum(
            self.observations['pheromones'].get('alarm', np.zeros(1)).flatten()
        )
        self.beliefs.danger_level = (
            0.9 * self.beliefs.danger_level + 
            0.1 * min(1.0, danger_signals)
        )
        
        # Update task urgency beliefs
        self._update_task_urgencies()
        
    def _update_task_urgencies(self) -> None:
        """Update beliefs about task urgencies."""
        # Foraging urgency
        self.beliefs.task_urgency[TaskType.FORAGING] = (
            1.0 - self.energy / self.config['physical']['energy']['initial']
        )
        
        # Defense urgency
        self.beliefs.task_urgency[TaskType.DEFENSE] = self.beliefs.danger_level
        
        # Other task urgencies could be updated based on observations
        # and pheromone concentrations
        
    def _select_actions(self) -> None:
        """Select actions using active inference."""
        # Compute expected free energy for each possible action
        task_energies = self._compute_task_energies()
        
        # Select task with lowest expected free energy
        if np.random.random() > self.exploration_rate:
            self.current_task = min(task_energies.items(), key=lambda x: x[1])[0]
            
        # Execute task-specific behavior
        self._execute_task_behavior()
        
    def _compute_task_energies(self) -> Dict[TaskType, float]:
        """Compute expected free energy for each task."""
        energies = {}
        
        for task in TaskType:
            # Prior preference term
            prior = -np.log(self.preferences[task])
            
            # Epistemic value term (information gain)
            epistemic = -self.beliefs.task_urgency[task]
            
            # Pragmatic value term (expected utility)
            pragmatic = self._compute_pragmatic_value(task)
            
            # Combine terms
            energies[task] = prior + epistemic + pragmatic
            
        return energies
        
    def _compute_pragmatic_value(self, task: TaskType) -> float:
        """Compute pragmatic value of a task."""
        if task == TaskType.FORAGING:
            return -self.energy / self.config['physical']['energy']['initial']
        elif task == TaskType.DEFENSE:
            return self.beliefs.danger_level
        # Add other task-specific values
        return 0.0
        
    def _execute_task_behavior(self) -> None:
        """Execute behavior for current task."""
        if self.current_task == TaskType.FORAGING:
            self._forage()
        elif self.current_task == TaskType.EXPLORATION:
            self._explore()
        elif self.current_task == TaskType.DEFENSE:
            self._defend()
        # Add other task behaviors
        
    def _forage(self) -> None:
        """Execute foraging behavior."""
        if self.carrying:
            # Return to nest
            self._move_towards(self.beliefs.nest_location)
            self._deposit_pheromone('home', 0.1)
        elif self.beliefs.food_location:
            # Move towards food
            self._move_towards(self.beliefs.food_location)
            self._deposit_pheromone('food', 0.1)
        else:
            # Random search
            self._random_walk()
            
    def _explore(self) -> None:
        """Execute exploration behavior."""
        # Implement levy flight or other exploration pattern
        if np.random.random() < 0.1:
            self.orientation += np.random.normal(0, np.pi/4)
        self.speed = self.config['physical']['max_speed']
        
    def _defend(self) -> None:
        """Execute defense behavior."""
        if self.beliefs.danger_level > 0.5:
            # Move towards danger and deposit alarm pheromone
            self._deposit_pheromone('alarm', 0.2)
            
    def _move_towards(self, target: Position) -> None:
        """Move towards a target position."""
        if target and self.position:
            dx = target.x - self.position.x
            dy = target.y - self.position.y
            target_orientation = np.arctan2(dy, dx)
            
            # Update orientation
            angle_diff = (target_orientation - self.orientation + np.pi) % (2*np.pi) - np.pi
            max_turn = self.config['physical']['turn_rate']
            self.orientation += np.clip(angle_diff, -max_turn, max_turn)
            
            # Update speed
            self.speed = self.config['physical']['max_speed']
            
    def _random_walk(self) -> None:
        """Execute random walk behavior."""
        if np.random.random() < 0.1:
            self.orientation += np.random.normal(0, np.pi/4)
        self.speed = self.config['physical']['max_speed'] * 0.5
        
    def _deposit_pheromone(self, ptype: str, amount: float) -> None:
        """Deposit pheromone at current position."""
        pass  # This will be implemented by the environment
        
    def _distance_to(self, target: Position) -> float:
        """Calculate distance to target position."""
        if not self.position or not target:
            return float('inf')
        dx = target.x - self.position.x
        dy = target.y - self.position.y
        return np.sqrt(dx*dx + dy*dy) 