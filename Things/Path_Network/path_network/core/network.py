"""
Network topology management for the Path Network simulation.
Handles the creation and management of the network of Active Inference agents.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .agent import ActiveInferenceAgent, AgentConfig

@dataclass
class NetworkConfig:
    """Configuration parameters for the network topology."""
    num_nodes: int = 10
    initial_connectivity: float = 0.3  # Probability of edge creation
    min_weight: float = 0.1
    max_weight: float = 1.0
    dynamic_topology: bool = True
    topology_update_interval: int = 100  # Steps between topology updates

class PathNetwork:
    """
    Manages a network of Active Inference agents and their interactions.
    
    The network topology is represented as a weighted directed graph where
    edges represent influence relationships between agents.
    """
    
    def __init__(self, network_config: NetworkConfig):
        self.config = network_config
        self.graph = nx.DiGraph()
        self.agents: Dict[int, ActiveInferenceAgent] = {}
        self.step_counter = 0
        
        # Initialize network
        self._initialize_network()
        
    def _initialize_network(self) -> None:
        """Initialize the network topology and create agents."""
        # Create nodes with agents
        for i in range(self.config.num_nodes):
            agent_config = AgentConfig(
                initial_height=np.random.normal(0, 0.1)
            )
            self.agents[i] = ActiveInferenceAgent(agent_config)
            self.graph.add_node(i)
        
        # Create random edges
        for i in range(self.config.num_nodes):
            for j in range(self.config.num_nodes):
                if i != j and np.random.random() < self.config.initial_connectivity:
                    weight = np.random.uniform(
                        self.config.min_weight,
                        self.config.max_weight
                    )
                    self.graph.add_edge(i, j, weight=weight)
    
    def update_topology(self) -> None:
        """
        Update network topology based on agent interactions and performance.
        This is called periodically to evolve the network structure.
        """
        if not self.config.dynamic_topology:
            return
            
        # Update edge weights based on prediction error correlation
        for i, j in self.graph.edges():
            agent_i = self.agents[i]
            agent_j = self.agents[j]
            
            # Compute correlation between agents' prediction errors
            corr = np.corrcoef(
                agent_i.prediction_error_history[-100:],
                agent_j.prediction_error_history[-100:]
            )[0, 1]
            
            # Update edge weight
            new_weight = np.clip(
                abs(corr),
                self.config.min_weight,
                self.config.max_weight
            )
            self.graph[i][j]['weight'] = new_weight
            
        # Potentially add or remove edges based on agent performance
        self._update_connections()
    
    def _update_connections(self) -> None:
        """Update network connections based on agent performance."""
        # Add edges between well-performing agents
        for i in range(self.config.num_nodes):
            for j in range(self.config.num_nodes):
                if i != j and not self.graph.has_edge(i, j):
                    # Add edge if agents have similar performance
                    if self._should_connect(i, j):
                        weight = np.random.uniform(
                            self.config.min_weight,
                            self.config.max_weight
                        )
                        self.graph.add_edge(i, j, weight=weight)
        
        # Remove edges with very low weights
        edges_to_remove = [
            (i, j) for i, j, w in self.graph.edges(data='weight')
            if w < self.config.min_weight
        ]
        self.graph.remove_edges_from(edges_to_remove)
    
    def _should_connect(self, i: int, j: int) -> bool:
        """Determine if two agents should be connected based on their performance."""
        agent_i = self.agents[i]
        agent_j = self.agents[j]
        
        # Compare recent performance
        errors_i = np.mean(agent_i.prediction_error_history[-50:])
        errors_j = np.mean(agent_j.prediction_error_history[-50:])
        
        # Connect if agents have similar error levels
        return abs(errors_i - errors_j) < 0.1
    
    def step(self, global_water_level: float) -> Dict[int, float]:
        """
        Perform one step of network simulation.
        
        Args:
            global_water_level: The current global water level
            
        Returns:
            Dict mapping node IDs to their new heights
        """
        self.step_counter += 1
        
        # Update agent states
        new_heights = {}
        for node_id, agent in self.agents.items():
            # Compute weighted average of neighbors' heights
            neighbor_influence = 0.0
            total_weight = 0.0
            
            for neighbor_id in self.graph.predecessors(node_id):
                weight = self.graph[neighbor_id][node_id]['weight']
                neighbor_height = self.agents[neighbor_id].height
                neighbor_influence += weight * neighbor_height
                total_weight += weight
            
            if total_weight > 0:
                effective_level = (
                    0.7 * global_water_level +
                    0.3 * (neighbor_influence / total_weight)
                )
            else:
                effective_level = global_water_level
            
            # Update agent
            new_height = agent.act(effective_level)
            new_heights[node_id] = new_height
        
        # Periodically update topology
        if (self.step_counter % self.config.topology_update_interval) == 0:
            self.update_topology()
        
        return new_heights
    
    def get_network_state(self) -> Tuple[nx.DiGraph, Dict[int, float]]:
        """
        Get the current state of the network.
        
        Returns:
            Tuple containing the network graph and current agent heights
        """
        heights = {i: agent.height for i, agent in self.agents.items()}
        return self.graph, heights 