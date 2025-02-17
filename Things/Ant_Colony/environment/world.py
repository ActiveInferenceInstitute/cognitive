"""
World Environment Implementation

This module implements the environment for the ant colony simulation,
including terrain generation, resource management, and physics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import noise  # For Perlin noise terrain generation
from scipy.ndimage import gaussian_filter

@dataclass
class Position:
    """2D position with optional orientation."""
    x: float
    y: float
    theta: float = 0.0

@dataclass
class Resource:
    """Resource entity in the environment."""
    position: Position
    type: str
    amount: float
    energy: float
    decay_rate: float

class TerrainGenerator:
    """Generates and manages terrain features."""
    
    def __init__(self, config: dict):
        """Initialize terrain generator."""
        self.config = config
        self.size = config['world']['size']
        self.resolution = config['world']['resolution']
        
        # Initialize terrain grid
        self.height_map = np.zeros(self.size)
        self.friction_map = np.zeros(self.size)
        self.type_map = np.zeros(self.size, dtype=str)
        
        # Generate initial terrain
        self._generate_terrain()
        
    def _generate_terrain(self):
        """Generate terrain using Perlin noise."""
        # Generate base height map
        scale = self.config['terrain']['generation']['scale']
        octaves = self.config['terrain']['generation']['octaves']
        seed = self.config['terrain']['generation']['seed']
        
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                self.height_map[i, j] = noise.pnoise2(
                    i/scale, 
                    j/scale, 
                    octaves=octaves, 
                    persistence=0.5, 
                    lacunarity=2.0, 
                    base=seed
                )
        
        # Normalize height map
        self.height_map = (self.height_map - self.height_map.min()) / (self.height_map.max() - self.height_map.min())
        
        # Generate terrain types and friction based on height
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                height = self.height_map[i, j]
                if height < 0.3:
                    self.type_map[i, j] = "sand"
                    self.friction_map[i, j] = self.config['terrain']['types'][2]['friction']
                elif height < 0.7:
                    self.type_map[i, j] = "soil"
                    self.friction_map[i, j] = self.config['terrain']['types'][0]['friction']
                else:
                    self.type_map[i, j] = "rock"
                    self.friction_map[i, j] = self.config['terrain']['types'][1]['friction']
                    
        # Add obstacles
        self._add_obstacles()
        
    def _add_obstacles(self):
        """Add obstacles to the terrain."""
        density = self.config['terrain']['features']['obstacles']['density']
        min_size = self.config['terrain']['features']['obstacles']['min_size']
        max_size = self.config['terrain']['features']['obstacles']['max_size']
        
        num_obstacles = int(density * self.size[0] * self.size[1])
        
        for _ in range(num_obstacles):
            # Random position and size
            pos_x = np.random.randint(0, self.size[0])
            pos_y = np.random.randint(0, self.size[1])
            size_x = np.random.randint(min_size[0], max_size[0] + 1)
            size_y = np.random.randint(min_size[1], max_size[1] + 1)
            
            # Add obstacle
            x_range = slice(max(0, pos_x), min(self.size[0], pos_x + size_x))
            y_range = slice(max(0, pos_y), min(self.size[1], pos_y + size_y))
            
            self.height_map[x_range, y_range] = 1.0
            self.type_map[x_range, y_range] = "rock"
            self.friction_map[x_range, y_range] = self.config['terrain']['types'][1]['friction']

class PheromoneGrid:
    """Manages pheromone diffusion and evaporation."""
    
    def __init__(self, config: dict):
        """Initialize pheromone grid."""
        self.config = config
        self.size = config['world']['size']
        self.resolution = config['pheromone_grid']['resolution']
        
        # Initialize pheromone layers
        self.layers = {}
        for layer in config['pheromone_grid']['layers']:
            self.layers[layer['name']] = {
                'grid': np.zeros(self.size),
                'diffusion_rate': layer['diffusion_rate'],
                'evaporation_rate': layer['evaporation_rate']
            }
            
    def update(self, dt: float):
        """Update pheromone concentrations."""
        for layer_name, layer in self.layers.items():
            # Diffusion
            layer['grid'] = gaussian_filter(
                layer['grid'],
                sigma=layer['diffusion_rate'] * dt
            )
            
            # Evaporation
            layer['grid'] *= (1 - layer['evaporation_rate'] * dt)
            
            # Enforce bounds
            layer['grid'] = np.clip(
                layer['grid'],
                self.config['pheromone_grid']['dynamics']['min_value'],
                self.config['pheromone_grid']['dynamics']['max_value']
            )
            
    def deposit(self, position: Position, pheromone_type: str, amount: float):
        """Deposit pheromone at specified position."""
        if pheromone_type not in self.layers:
            return
            
        # Convert position to grid coordinates
        x = int(position.x / self.resolution)
        y = int(position.y / self.resolution)
        
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            self.layers[pheromone_type]['grid'][x, y] += amount
            
    def get_concentration(self, position: Position, pheromone_type: str) -> float:
        """Get pheromone concentration at specified position."""
        if pheromone_type not in self.layers:
            return 0.0
            
        x = int(position.x / self.resolution)
        y = int(position.y / self.resolution)
        
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            return self.layers[pheromone_type]['grid'][x, y]
        return 0.0

class World:
    """Main environment class managing all environmental components."""
    
    def __init__(self, config: dict):
        """Initialize world environment."""
        self.config = config
        self.size = config['world']['size']
        self.resolution = config['world']['resolution']
        
        # Initialize components
        self.terrain = TerrainGenerator(config)
        self.pheromones = PheromoneGrid(config)
        
        # Resource management
        self.resources: List[Resource] = []
        self._initialize_resources()
        
        # Time tracking
        self.time = 0.0
        self.day_time = 0.0
        
        # Environmental conditions
        self.temperature = config['conditions']['temperature']['base']
        self.humidity = config['conditions']['humidity']['base']
        self.light = config['conditions']['light']['base']
        
    def step(self, dt: float):
        """Update world state."""
        # Update time
        self.time += dt
        self.day_time = (self.day_time + dt) % self.config['time']['cycles']['day_length']
        
        # Update environmental conditions
        self._update_conditions()
        
        # Update resources
        self._update_resources(dt)
        
        # Update pheromones
        self.pheromones.update(dt)
        
    def _initialize_resources(self):
        """Initialize resource distribution."""
        # Food resources
        food_config = self.config['resources']['food']
        
        if food_config['distribution']['method'] == "clustered":
            self._create_resource_clusters(
                food_config['distribution']['cluster_size'],
                food_config['distribution']['total_amount']
            )
        else:
            self._create_random_resources(
                food_config['distribution']['total_amount']
            )
            
        # Water resources
        water_config = self.config['resources']['water']
        self._create_random_resources(
            water_config['distribution']['total_amount'],
            resource_type="water"
        )
        
    def _create_resource_clusters(self, cluster_size: int, total_amount: float):
        """Create clustered resource distribution."""
        num_clusters = int(total_amount / cluster_size)
        
        for _ in range(num_clusters):
            # Random cluster center
            center_x = np.random.uniform(0, self.size[0])
            center_y = np.random.uniform(0, self.size[1])
            
            # Create resources in cluster
            for _ in range(cluster_size):
                # Random offset from center
                offset_x = np.random.normal(0, self.config['resources']['food']['distribution']['cluster_spread'])
                offset_y = np.random.normal(0, self.config['resources']['food']['distribution']['cluster_spread'])
                
                x = np.clip(center_x + offset_x, 0, self.size[0] - 1)
                y = np.clip(center_y + offset_y, 0, self.size[1] - 1)
                
                # Create resource
                resource = Resource(
                    position=Position(x, y),
                    type="food",
                    amount=np.random.uniform(1.0, 3.0),
                    energy=10.0,
                    decay_rate=0.001
                )
                self.resources.append(resource)
                
    def _create_random_resources(self, total_amount: float, resource_type: str = "food"):
        """Create randomly distributed resources."""
        num_resources = int(total_amount)
        
        for _ in range(num_resources):
            x = np.random.uniform(0, self.size[0])
            y = np.random.uniform(0, self.size[1])
            
            resource = Resource(
                position=Position(x, y),
                type=resource_type,
                amount=np.random.uniform(1.0, 3.0),
                energy=5.0 if resource_type == "water" else 10.0,
                decay_rate=0.001
            )
            self.resources.append(resource)
            
    def _update_resources(self, dt: float):
        """Update resource states."""
        # Update existing resources
        for resource in self.resources[:]:
            resource.amount -= resource.decay_rate * dt
            if resource.amount <= 0:
                self.resources.remove(resource)
                
        # Respawn resources if needed
        if len([r for r in self.resources if r.type == "food"]) < self.config['resources']['food']['distribution']['total_amount'] * 0.5:
            self._create_random_resources(10)
            
    def _update_conditions(self):
        """Update environmental conditions."""
        # Day/night cycle
        day_progress = self.day_time / self.config['time']['cycles']['day_length']
        
        # Temperature variation
        self.temperature = self.config['conditions']['temperature']['base'] + \
                         self.config['conditions']['temperature']['variation'] * \
                         np.sin(2 * np.pi * day_progress)
                         
        # Humidity variation
        self.humidity = self.config['conditions']['humidity']['base'] + \
                      self.config['conditions']['humidity']['variation'] * \
                      np.sin(2 * np.pi * day_progress + np.pi/2)
                      
        # Light variation
        self.light = self.config['conditions']['light']['base'] + \
                    self.config['conditions']['light']['variation'] * \
                    np.sin(2 * np.pi * day_progress)
                    
    def get_state(self, position: Position) -> Dict:
        """Get environmental state at specified position."""
        return {
            'terrain': {
                'height': self._get_height(position),
                'type': self._get_terrain_type(position),
                'friction': self._get_friction(position)
            },
            'pheromones': {
                name: self.pheromones.get_concentration(position, name)
                for name in self.pheromones.layers.keys()
            },
            'resources': self._get_nearby_resources(position),
            'conditions': {
                'temperature': self.temperature,
                'humidity': self.humidity,
                'light': self.light
            }
        }
        
    def _get_height(self, position: Position) -> float:
        """Get terrain height at position."""
        x = int(position.x / self.resolution)
        y = int(position.y / self.resolution)
        
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            return self.terrain.height_map[x, y]
        return 0.0
        
    def _get_terrain_type(self, position: Position) -> str:
        """Get terrain type at position."""
        x = int(position.x / self.resolution)
        y = int(position.y / self.resolution)
        
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            return self.terrain.type_map[x, y]
        return "none"
        
    def _get_friction(self, position: Position) -> float:
        """Get terrain friction at position."""
        x = int(position.x / self.resolution)
        y = int(position.y / self.resolution)
        
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            return self.terrain.friction_map[x, y]
        return 1.0
        
    def _get_nearby_resources(self, position: Position, radius: float = 5.0) -> List[Resource]:
        """Get resources within specified radius of position."""
        nearby = []
        for resource in self.resources:
            dx = resource.position.x - position.x
            dy = resource.position.y - position.y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance <= radius:
                nearby.append(resource)
                
        return nearby 