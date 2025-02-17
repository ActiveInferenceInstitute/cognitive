"""
World environment module for ant colony simulation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import noise

@dataclass
class Position:
    """Represents a position in 2D space with orientation."""
    x: float
    y: float
    theta: float = 0.0

@dataclass
class Resource:
    """Represents a resource in the environment."""
    position: Position
    type: str
    amount: float
    max_amount: float

class World:
    """Represents the physical environment for the ant colony simulation."""
    
    def __init__(self, config: dict):
        """Initialize the world environment."""
        self.config = config
        self.size = config['size']
        
        # Initialize terrain
        self.terrain = self._generate_terrain()
        
        # Initialize resources
        self.resources: Dict[str, List[Resource]] = {
            'food': [],
            'water': [],
            'building_materials': []
        }
        self._initialize_resources()
        
        # Initialize pheromone layers
        self.pheromones = {
            'food': np.zeros(self.size),
            'home': np.zeros(self.size),
            'alarm': np.zeros(self.size),
            'trail': np.zeros(self.size)
        }
        
    def _generate_terrain(self) -> np.ndarray:
        """Generate terrain using Perlin noise."""
        terrain = np.zeros(self.size)
        scale = self.config.get('terrain_scale', 50.0)
        octaves = self.config.get('terrain_octaves', 6)
        persistence = self.config.get('terrain_persistence', 0.5)
        lacunarity = self.config.get('terrain_lacunarity', 2.0)
        
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                terrain[i, j] = noise.pnoise2(i/scale, 
                                            j/scale, 
                                            octaves=octaves,
                                            persistence=persistence,
                                            lacunarity=lacunarity,
                                            repeatx=self.size[0],
                                            repeaty=self.size[1],
                                            base=42)
        
        # Normalize to [0, 1]
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
        return terrain
        
    def _initialize_resources(self):
        """Initialize resources in the environment."""
        # Add food sources
        for _ in range(self.config['food_sources']['count']):
            pos = self._random_position()
            amount = np.random.uniform(*self.config['food_sources']['value_range'])
            size = np.random.uniform(*self.config['food_sources']['size_range'])
            
            resource = Resource(
                position=pos,
                type='food',
                amount=amount,
                max_amount=amount
            )
            self.resources['food'].append(resource)
            
    def _random_position(self) -> Position:
        """Generate a random position within the world bounds."""
        x = np.random.uniform(0, self.size[0])
        y = np.random.uniform(0, self.size[1])
        theta = np.random.uniform(0, 2 * np.pi)
        return Position(x, y, theta)
        
    def update(self, dt: float):
        """Update world state."""
        # Update pheromone diffusion and evaporation
        self._update_pheromones(dt)
        
        # Update resource regeneration
        self._update_resources(dt)
        
    def _update_pheromones(self, dt: float):
        """Update pheromone diffusion and evaporation."""
        for ptype, grid in self.pheromones.items():
            # Apply diffusion
            grid = self._diffuse(grid, dt)
            
            # Apply evaporation
            decay = self.config.get('pheromone_decay', 0.995)
            grid *= decay ** dt
            
            self.pheromones[ptype] = grid
            
    def _diffuse(self, grid: np.ndarray, dt: float) -> np.ndarray:
        """Apply diffusion to a pheromone grid."""
        diffusion_rate = 0.1 * dt
        return grid + diffusion_rate * (
            np.roll(grid, 1, axis=0) + 
            np.roll(grid, -1, axis=0) + 
            np.roll(grid, 1, axis=1) + 
            np.roll(grid, -1, axis=1) - 
            4 * grid
        )
        
    def _update_resources(self, dt: float):
        """Update resource regeneration."""
        for resource_type, resources in self.resources.items():
            for resource in resources:
                if resource.amount < resource.max_amount:
                    # Apply regeneration
                    regen_rate = self.config.get('resource_regeneration_rate', 0.1)
                    resource.amount = min(
                        resource.amount + regen_rate * dt,
                        resource.max_amount
                    )
                    
    def get_state(self) -> dict:
        """Get the current state of the world."""
        return {
            'pheromones': self.pheromones,
            'resources': [resource for resources in self.resources.values() for resource in resources],
            'terrain': self.terrain
        }
        
    def get_local_state(self, position: Position, radius: float) -> dict:
        """Get the local state around a position within given radius."""
        x, y = position.x, position.y
        
        # Get local pheromone values
        local_pheromones = {}
        for ptype, grid in self.pheromones.items():
            x_min = max(0, int(x - radius))
            x_max = min(self.size[0], int(x + radius + 1))
            y_min = max(0, int(y - radius))
            y_max = min(self.size[1], int(y + radius + 1))
            
            local_pheromones[ptype] = grid[x_min:x_max, y_min:y_max]
            
        # Get nearby resources
        nearby_resources = []
        for resources in self.resources.values():
            for resource in resources:
                dx = resource.position.x - x
                dy = resource.position.y - y
                if np.sqrt(dx*dx + dy*dy) <= radius:
                    nearby_resources.append(resource)
                    
        return {
            'pheromones': local_pheromones,
            'resources': nearby_resources,
            'terrain': self._get_local_terrain(position, radius)
        }
        
    def _get_local_terrain(self, position: Position, radius: float) -> np.ndarray:
        """Get local terrain heights around a position."""
        x, y = position.x, position.y
        x_min = max(0, int(x - radius))
        x_max = min(self.size[0], int(x + radius + 1))
        y_min = max(0, int(y - radius))
        y_max = min(self.size[1], int(y + radius + 1))
        
        return self.terrain[x_min:x_max, y_min:y_max]
        
    def deposit_pheromone(self, position: Position, ptype: str, amount: float):
        """Deposit pheromone at a position."""
        x, y = int(position.x), int(position.y)
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            self.pheromones[ptype][x, y] += amount
            
    def remove_resource(self, resource: Resource, amount: float) -> float:
        """Remove an amount from a resource, returns actual amount removed."""
        if resource.amount >= amount:
            resource.amount -= amount
            return amount
        else:
            removed = resource.amount
            resource.amount = 0
            return removed 