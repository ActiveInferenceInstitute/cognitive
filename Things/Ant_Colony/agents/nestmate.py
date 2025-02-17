"""
Nestmate Agent Implementation

This module implements a simplified version of the Nestmate agent class,
representing an individual ant in the colony.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass

class TaskType(Enum):
    """Possible task types for a Nestmate agent."""
    FORAGING = "foraging"
    MAINTENANCE = "maintenance"
    NURSING = "nursing"
    DEFENSE = "defense"
    EXPLORATION = "exploration"

@dataclass
class Position:
    """2D position with orientation."""
    x: float
    y: float
    theta: float = 0.0

class Nestmate:
    """Individual ant agent with basic behaviors."""
    
    def __init__(self, config: dict):
        """Initialize Nestmate agent."""
        self.config = config
        
        # Physical state
        self.position = Position(0.0, 0.0, 0.0)
        self.velocity = np.zeros(2)
        self.energy = config['physical']['energy']['initial']
        
        # Task state
        self.current_task = TaskType.EXPLORATION
        self.carrying = None
        
        # Sensors
        self.sensor_range = config['physical']['sensor_range']
        
        # Movement parameters
        self.max_speed = config['physical']['max_speed']
        self.turn_rate = config['physical']['turn_rate']
        
    def sense(self, world_state: dict) -> dict:
        """Process sensory inputs from environment."""
        # Get nearby entities within sensor range
        nearby = {
            'food': [],
            'nestmates': [],
            'obstacles': [],
            'pheromones': {}
        }
        
        # Process food sources
        for food in world_state['resources']:
            dist = self._distance_to(food.position)
            if dist <= self.sensor_range:
                nearby['food'].append((food, dist))
        
        # Process other agents
        for agent in world_state['agents']:
            if agent != self:
                dist = self._distance_to(agent.position)
                if dist <= self.sensor_range:
                    nearby['nestmates'].append((agent, dist))
        
        # Process pheromones
        for p_type, value in world_state['pheromones'].items():
            nearby['pheromones'][p_type] = value
            
        return nearby
        
    def decide_action(self, sensed: dict) -> tuple:
        """Decide next action based on current state and sensory input."""
        # Default behavior: random walk
        speed = self.max_speed
        turn = np.random.uniform(-self.turn_rate, self.turn_rate)
        
        # Task-specific behaviors
        if self.current_task == TaskType.FORAGING:
            # If carrying food, head back to nest
            if self.carrying:
                turn = self._angle_to_nest()
            # Otherwise, follow food pheromones or explore
            elif 'food' in sensed['pheromones']:
                pheromone_gradient = sensed['pheromones']['food']
                if np.any(pheromone_gradient > 0):
                    turn = self._follow_gradient(pheromone_gradient)
                    
        elif self.current_task == TaskType.EXPLORATION:
            # Random walk with longer persistence
            if np.random.random() < 0.1:  # 10% chance to change direction
                turn = np.random.uniform(-np.pi, np.pi)
                
        return speed, turn
        
    def update(self, dt: float, world_state: dict):
        """Update agent state."""
        # Sense environment
        sensed = self.sense(world_state)
        
        # Decide action
        speed, turn = self.decide_action(sensed)
        
        # Update position and orientation
        self.position.theta += turn * dt
        self.position.theta = self.position.theta % (2 * np.pi)
        
        dx = speed * np.cos(self.position.theta) * dt
        dy = speed * np.sin(self.position.theta) * dt
        
        self.position.x += dx
        self.position.y += dy
        
        # Update energy
        self.energy -= self.config['physical']['energy']['consumption_rate'] * dt
        if self.carrying:
            self.energy -= self.config['physical']['energy']['consumption_rate'] * dt
            
        # Consider task switching
        self._consider_task_switch(sensed)
        
    def _distance_to(self, other_pos: Position) -> float:
        """Calculate distance to another position."""
        dx = other_pos.x - self.position.x
        dy = other_pos.y - self.position.y
        return np.sqrt(dx*dx + dy*dy)
        
    def _angle_to_nest(self) -> float:
        """Calculate turn angle towards nest."""
        # Simplified: assume nest is at (0,0)
        dx = -self.position.x
        dy = -self.position.y
        target_angle = np.arctan2(dy, dx)
        current_angle = self.position.theta
        
        # Calculate shortest turn
        diff = target_angle - current_angle
        while diff > np.pi:
            diff -= 2*np.pi
        while diff < -np.pi:
            diff += 2*np.pi
            
        return np.clip(diff, -self.turn_rate, self.turn_rate)
        
    def _follow_gradient(self, gradient: np.ndarray) -> float:
        """Calculate turn angle to follow a pheromone gradient."""
        # Simplified: assume gradient gives us desired direction
        target_angle = np.arctan2(gradient[1], gradient[0])
        return self._angle_to_nest()  # Reuse angle calculation
        
    def _consider_task_switch(self, sensed: dict):
        """Consider switching current task."""
        # Simple task switching based on energy and random chance
        if self.energy < self.config['physical']['energy']['critical_level']:
            self.current_task = TaskType.FORAGING
        elif np.random.random() < 0.001:  # 0.1% chance to switch tasks
            available_tasks = list(TaskType)
            available_tasks.remove(self.current_task)
            self.current_task = np.random.choice(available_tasks) 