"""
Colony Management System

This module implements the colony management system that coordinates
the ant colony simulation, including agent management, task allocation,
and collective behavior.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import networkx as nx
from ant_colony.agents.nestmate import Nestmate, TaskType
from ant_colony.environment.world import World, Position, Resource

@dataclass
class ColonyStats:
    """Container for colony statistics."""
    population: int
    task_distribution: Dict[TaskType, int]
    resource_levels: Dict[str, float]
    efficiency_metrics: Dict[str, float]
    coordination_metrics: Dict[str, float]

class Colony:
    """
    Colony management system coordinating multiple Nestmate agents.
    """
    
    def __init__(self, config: dict, environment: World):
        """Initialize colony."""
        self.config = config
        self.environment = environment
        
        # Initialize colony state
        self.nest_position = Position(
            x=environment.config['nest_location'][0],
            y=environment.config['nest_location'][1],
            theta=0.0
        )
        
        # Initialize resources
        self.resources = {
            'food': 0.0,
            'water': 0.0,
            'building_materials': 0.0
        }
        
        # Initialize task needs and allocation
        self.task_needs = {task: 0.0 for task in TaskType}
        self.task_allocation = {task: set() for task in TaskType}
        
        # Initialize agents
        self.agents: List[Nestmate] = []
        self._initialize_agents()
        
        # Initialize social network
        self.interaction_network = nx.Graph()
        for agent in self.agents:
            self.interaction_network.add_node(agent)
            
        # Initialize statistics
        self.stats = self._compute_stats()
        
    def _initialize_agents(self) -> None:
        """Initialize colony agents."""
        num_agents = self.config['initial_population']
        
        for _ in range(num_agents):
            # Create agent with random position near nest
            pos_x = self.nest_position.x + np.random.normal(0, 2.0)
            pos_y = self.nest_position.y + np.random.normal(0, 2.0)
            pos = Position(pos_x, pos_y, np.random.uniform(0, 2*np.pi))
            
            # Create agent with agent-specific config
            agent_config = {
                'physical': {
                    'sensor_range': 10.0,
                    'max_speed': 2.0,
                    'turn_rate': 0.5,
                    'energy': {
                        'initial': 100.0,
                        'critical_level': 30.0,
                        'consumption_rate': 0.1
                    }
                },
                'behavior': {
                    'task_switching': {
                        'flexibility': 0.01
                    },
                    'learning_rate': 0.1,
                    'exploration_rate': 0.2
                }
            }
            
            agent = Nestmate(agent_config)
            agent.position = pos
            
            # Initialize beliefs after creating agent
            agent.beliefs.nest_location = self.nest_position
            
            self.agents.append(agent)
            
        # Distribute initial tasks
        self._distribute_tasks()
        
    def _distribute_tasks(self) -> None:
        """Distribute initial tasks among agents."""
        num_agents = len(self.agents)
        
        # Calculate number of agents per task based on config
        task_counts = {
            TaskType.FORAGING: int(0.4 * num_agents),  # 40% foragers
            TaskType.MAINTENANCE: int(0.2 * num_agents),  # 20% maintenance
            TaskType.NURSING: int(0.2 * num_agents),  # 20% nurses
            TaskType.DEFENSE: int(0.1 * num_agents),  # 10% defenders
            TaskType.EXPLORATION: int(0.1 * num_agents)  # 10% explorers
        }
        
        # Assign tasks
        agents_copy = self.agents.copy()
        np.random.shuffle(agents_copy)
        
        current_idx = 0
        for task, count in task_counts.items():
            for _ in range(count):
                if current_idx < len(agents_copy):
                    agent = agents_copy[current_idx]
                    agent.current_task = task
                    current_idx += 1
                    
    def update(self, dt: float) -> None:
        """Update colony state."""
        # Get current world state
        world_state = {
            'agents': self.agents,
            'resources': self.environment.resources,
            'pheromones': self.environment.pheromones
        }
        
        # Update each agent
        for agent in self.agents:
            agent.update(dt, world_state)
            
        # Update social network
        self._update_social_network()
        
        # Update resources
        self._update_resources(dt)
        
        # Update statistics
        self.stats = self._compute_stats()
        
    def _update_social_network(self) -> None:
        """Update agent interaction network."""
        # Clear old edges
        self.interaction_network.clear_edges()
        
        # Add edges between nearby agents
        for i, agent1 in enumerate(self.agents):
            for agent2 in self.agents[i+1:]:
                if self._distance_between(agent1.position, agent2.position) < 5.0:
                    self.interaction_network.add_edge(agent1, agent2)
                    
    def _update_resources(self, dt: float) -> None:
        """Update colony's resource levels."""
        # Apply consumption
        consumption_rate = self.config.get('resource_consumption_rate', 0.1)
        num_agents = len(self.agents)
        
        self.resources['food'] = max(0.0, 
            self.resources['food'] - consumption_rate * num_agents * dt)
        self.resources['water'] = max(0.0,
            self.resources['water'] - consumption_rate * num_agents * dt * 0.5)
            
    def _compute_stats(self) -> ColonyStats:
        """Compute current colony statistics."""
        # Count agents per task
        task_dist = {task: 0 for task in TaskType}
        for agent in self.agents:
            task_dist[agent.current_task] += 1
            
        # Compute efficiency metrics
        efficiency = {
            'food_gathering': self._compute_food_efficiency(),
            'task_completion': self._compute_task_efficiency(),
            'energy_efficiency': self._compute_energy_efficiency()
        }
        
        # Compute coordination metrics
        coordination = {
            'task_specialization': self._calculate_specialization(),
            'social_network_density': nx.density(self.interaction_network),
            'task_balance': self._compute_task_balance(task_dist)
        }
        
        return ColonyStats(
            population=len(self.agents),
            task_distribution=task_dist,
            resource_levels=self.resources.copy(),
            efficiency_metrics=efficiency,
            coordination_metrics=coordination
        )
        
    def _compute_food_efficiency(self) -> float:
        """Compute food gathering efficiency."""
        foragers = sum(1 for a in self.agents if a.current_task == TaskType.FORAGING)
        if foragers == 0:
            return 0.0
        return self.resources['food'] / foragers
        
    def _compute_task_efficiency(self) -> float:
        """Compute overall task completion efficiency."""
        # Simplified metric based on resource levels and population
        return min(1.0, self.resources['food'] / (len(self.agents) * 10))
        
    def _compute_energy_efficiency(self) -> float:
        """Compute energy usage efficiency."""
        return np.mean([agent.energy for agent in self.agents])
        
    def _distance_between(self, pos1: Position, pos2: Position) -> float:
        """Calculate distance between two positions."""
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        return np.sqrt(dx*dx + dy*dy)
        
    def step(self, dt: float):
        """Update colony state."""
        # Update task needs
        self._update_task_needs()
        
        # Update agent states and actions
        self._update_agents(dt)
        
        # Update social network
        self._update_social_network()
        
        # Update colony resources
        self._update_resources(dt)
        
        # Update statistics
        self._update_statistics()
        
        # Check emergency conditions
        self._check_emergencies()
        
    def _update_task_needs(self):
        """Update colony's task needs based on current state."""
        # Reset needs
        for task in TaskType:
            self.task_needs[task] = 0.0
            
        # Calculate needs based on various factors
        
        # Foraging need based on food levels
        food_ratio = self.resources['food'] / self.config['nest']['resources']['food_capacity']
        self.task_needs[TaskType.FORAGING] = 1.0 - food_ratio
        
        # Maintenance need based on nest condition (placeholder)
        self.task_needs[TaskType.MAINTENANCE] = 0.5  # Could be based on actual nest damage
        
        # Nursing need (placeholder)
        self.task_needs[TaskType.NURSING] = 0.3  # Could be based on brood size
        
        # Defense need based on threats (placeholder)
        self.task_needs[TaskType.DEFENSE] = 0.2  # Could be based on detected threats
        
        # Exploration need
        explored_area = len(set((agent.position[0], agent.position[1]) for agent in self.agents))
        total_area = self.environment.size[0] * self.environment.size[1]
        self.task_needs[TaskType.EXPLORATION] = 1.0 - (explored_area / total_area)
        
    def _update_agents(self, dt: float):
        """Update all agents' states and actions."""
        for agent in self.agents:
            # Get environmental state at agent's position
            pos = Position(agent.position[0], agent.position[1])
            env_state = self.environment.get_state(pos)
            
            # Get nearby agents
            nearby_agents = self._get_nearby_agents(agent)
            
            # Update agent observations
            observations = self._prepare_observations(agent, env_state, nearby_agents)
            
            # Get agent's action
            action = agent.update(observations)
            
            # Apply action
            self._apply_action(agent, action, dt)
            
            # Handle resource collection/deposition
            self._handle_resource_interaction(agent)
            
    def _get_nearby_agents(self, agent: Nestmate, radius: float = 5.0) -> List[Nestmate]:
        """Get list of agents within specified radius."""
        nearby = []
        for other in self.agents:
            if other != agent:
                distance = np.linalg.norm(other.position - agent.position)
                if distance <= radius:
                    nearby.append(other)
        return nearby
        
    def _prepare_observations(self, 
                            agent: Nestmate, 
                            env_state: Dict, 
                            nearby_agents: List[Nestmate]) -> Dict:
        """Prepare observation data for agent."""
        # Convert nearby agents to observation format
        nearby_positions = np.array([other.position for other in nearby_agents])
        
        return {
            'pheromone': np.array(list(env_state['pheromones'].values())),
            'food': np.array([r.amount for r in env_state['resources'] if r.type == 'food']),
            'nestmates': nearby_positions if len(nearby_positions) > 0 else np.zeros((0, 2)),
            'obstacles': np.array([1.0 if env_state['terrain']['type'] == 'rock' else 0.0]),
            'nest': agent.position - np.array([self.nest_position.x, self.nest_position.y])
        }
        
    def _apply_action(self, agent: Nestmate, action: np.ndarray, dt: float):
        """Apply agent's action and update its state."""
        # Extract movement components
        speed = np.clip(action[0], 0, agent.config['physical']['max_speed'])
        turn_rate = np.clip(action[1], -agent.config['physical']['turn_rate'],
                                      agent.config['physical']['turn_rate'])
        
        # Update orientation
        agent.orientation += turn_rate * dt
        agent.orientation = agent.orientation % (2 * np.pi)
        
        # Update position
        direction = np.array([np.cos(agent.orientation), np.sin(agent.orientation)])
        new_position = agent.position + speed * direction * dt
        
        # Check for collision and apply if valid
        if self._is_valid_position(new_position):
            agent.position = new_position
            
    def _is_valid_position(self, position: np.ndarray) -> bool:
        """Check if position is valid (within bounds and not in obstacle)."""
        # Check bounds
        if not (0 <= position[0] < self.environment.size[0] and
                0 <= position[1] < self.environment.size[1]):
            return False
            
        # Check for obstacles
        pos = Position(position[0], position[1])
        env_state = self.environment.get_state(pos)
        if env_state['terrain']['type'] == 'rock':
            return False
            
        return True
        
    def _handle_resource_interaction(self, agent: Nestmate):
        """Handle agent's interaction with resources."""
        # Check if agent is at nest
        at_nest = np.linalg.norm(agent.position - np.array([self.nest_position.x, self.nest_position.y])) < 2.0
        
        if at_nest and agent.carrying is not None:
            # Deposit resource
            self.resources[agent.carrying.type] += agent.carrying.amount
            agent.carrying = None
            
        elif agent.carrying is None:
            # Try to pick up resource
            pos = Position(agent.position[0], agent.position[1])
            nearby_resources = self.environment._get_nearby_resources(pos, radius=1.0)
            
            if nearby_resources:
                resource = nearby_resources[0]
                agent.carrying = resource
                self.environment.resources.remove(resource)
                
    def _update_statistics(self):
        """Update colony statistics."""
        # Update basic stats
        self.stats.population = len(self.agents)
        
        # Update task distribution
        for task in TaskType:
            self.stats.task_distribution[task] = len([a for a in self.agents if a.current_task == task])
            
        # Update resource levels
        self.stats.resource_levels = self.resources.copy()
        
        # Calculate efficiency metrics
        self.stats.efficiency_metrics = {
            'resource_gathering': self._calculate_resource_efficiency(),
            'task_completion': self._calculate_task_efficiency(),
            'energy_efficiency': self._calculate_energy_efficiency()
        }
        
        # Calculate coordination metrics
        self.stats.coordination_metrics = {
            'network_density': nx.density(self.interaction_network),
            'clustering_coefficient': nx.average_clustering(self.interaction_network),
            'task_specialization': self._calculate_specialization()
        }
        
    def _calculate_resource_efficiency(self) -> float:
        """Calculate resource gathering efficiency."""
        if self.stats.task_distribution[TaskType.FORAGING] == 0:
            return 0.0
        return self.resources['food'] / self.stats.task_distribution[TaskType.FORAGING]
        
    def _calculate_task_efficiency(self) -> float:
        """Calculate overall task completion efficiency."""
        total_need = sum(self.task_needs.values())
        if total_need == 0:
            return 1.0
        
        need_satisfaction = sum(min(1.0, len(self.task_allocation[task]) / (need * len(self.agents)))
                              for task, need in self.task_needs.items())
        return need_satisfaction / len(self.task_needs)
        
    def _calculate_energy_efficiency(self) -> float:
        """Calculate colony's energy efficiency."""
        total_energy = sum(agent.energy for agent in self.agents)
        return total_energy / (len(self.agents) * self.config['physical']['energy']['initial'])
        
    def _calculate_specialization(self) -> float:
        """Calculate degree of task specialization."""
        if not self.agents:
            return 0.0
            
        # Calculate how long agents stick to their tasks
        total_switches = sum(1 for agent in self.agents 
                           if len(agent.memory['temporal']) > 1 
                           and agent.memory['temporal'][-1]['state'] != agent.memory['temporal'][-2]['state'])
        
        return 1.0 - (total_switches / len(self.agents))
        
    def _check_emergencies(self):
        """Check and respond to emergency conditions."""
        # Check resource levels
        for resource_type, amount in self.resources.items():
            if amount < self.config['emergency']['resources']['critical_threshold']:
                self._handle_resource_emergency(resource_type)
                
        # Check for threats (placeholder)
        if self._detect_threats():
            self._handle_threat_emergency()
            
    def _handle_resource_emergency(self, resource_type: str):
        """Handle critical resource shortage."""
        if self.config['emergency']['resources']['emergency_allocation']:
            # Reassign more agents to foraging
            num_new_foragers = int(len(self.agents) * 0.2)  # 20% of colony
            current_foragers = set(self.task_allocation[TaskType.FORAGING])
            
            for agent in self.agents:
                if len(current_foragers) >= num_new_foragers:
                    break
                if agent not in current_foragers and agent.current_task != TaskType.DEFENSE:
                    agent.current_task = TaskType.FORAGING
                    current_foragers.add(agent)
                    
    def _detect_threats(self) -> bool:
        """Detect potential threats to the colony."""
        # Placeholder for threat detection
        return False
        
    def _handle_threat_emergency(self):
        """Handle detected threats."""
        if self.config['emergency']['threats']['mobilization_rate'] > 0:
            # Reassign agents to defense
            num_defenders = int(len(self.agents) * self.config['emergency']['threats']['mobilization_rate'])
            current_defenders = set(self.task_allocation[TaskType.DEFENSE])
            
            for agent in self.agents:
                if len(current_defenders) >= num_defenders:
                    break
                if agent not in current_defenders:
                    agent.current_task = TaskType.DEFENSE
                    current_defenders.add(agent)
        
    def _compute_task_balance(self, task_dist: Dict[TaskType, int]) -> float:
        """Compute how well-balanced the task distribution is."""
        if not self.agents:
            return 0.0
        
        # Ideal distribution would be uniform across tasks
        ideal_per_task = len(self.agents) / len(TaskType)
        deviations = [abs(count - ideal_per_task) for count in task_dist.values()]
        max_deviation = len(self.agents)  # Worst case: all agents on one task
        
        # Return normalized score (1.0 = perfect balance, 0.0 = worst balance)
        return 1.0 - sum(deviations) / (2 * max_deviation) 