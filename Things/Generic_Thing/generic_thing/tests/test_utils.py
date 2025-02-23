"""Test utilities for Generic Thing."""

import os
import logging
import datetime
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from functools import wraps

from generic_thing.visualization import Visualizer

# Configure plotting style
sns.set_style('whitegrid')  # Using seaborn's style directly instead of plt.style

# Set up logging
logger = logging.getLogger(__name__)

# Initialize visualizer for consistent path handling
visualizer = Visualizer()

def log_test_case(func: Callable) -> Callable:
    """Decorator for logging test case execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Running test: {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"Test completed: {func.__name__}")
        return result
    return wrapper

def generate_test_data(size: int = 10) -> Dict[str, Any]:
    """Generate test data for visualization tests."""
    timesteps = np.arange(size)
    internal_states = np.random.rand(size, 3)  # 3 internal state variables
    external_states = np.random.rand(size, 2)  # 2 external state variables
    free_energies = np.random.rand(size)
    prior = np.random.rand(size, 3)  # Prior beliefs about internal states
    posterior = np.random.rand(size, 3)  # Posterior beliefs about internal states
    observations = np.random.rand(size, 2)  # Observations of external states
    
    # Generate Markov blanket states
    sensory_states = np.random.rand(size, 2)
    internal_states_mb = np.random.rand(size, 3)
    active_states = np.random.rand(size, 2)
    
    return {
        'internal_states': internal_states,
        'external_states': external_states,
        'free_energies': free_energies,
        'prior': prior,
        'posterior': posterior,
        'observations': observations,
        'timesteps': timesteps,
        'sensory_states': sensory_states,
        'internal_states_mb': internal_states_mb,
        'active_states': active_states
    }

def plot_thing_hierarchy(thing: Any, filename: str) -> None:
    """Plot the hierarchical structure of things."""
    logger.info("Plotting thing hierarchy")
    G = nx.DiGraph()
    
    def add_nodes(t):
        G.add_node(t.id, name=t.name)
        for child in t.children:
            G.add_edge(t.id, child.id)
            add_nodes(child)
    
    add_nodes(thing)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1500, arrowsize=20)
    
    labels = nx.get_node_attributes(G, 'name')
    nx.draw_networkx_labels(G, pos, labels)
    
    path = visualizer._ensure_viz_path(filename)
    plt.savefig(path)
    plt.close()

def plot_free_energy_history(history: List[float], filename: str) -> None:
    """Plot the history of free energy values."""
    logger.info("Plotting free energy history")
    plt.figure(figsize=(10, 6))
    plt.plot(history, '-b', label='Free Energy')
    plt.title('Free Energy History')
    plt.xlabel('Update Step')
    plt.ylabel('Free Energy')
    plt.grid(True)
    plt.legend()
    
    path = visualizer._ensure_viz_path(filename)
    plt.savefig(path)
    plt.close()

def plot_message_network(message_passing: Any, filename: str) -> None:
    """Plot the message passing network."""
    logger.info("Plotting message network")
    G = nx.Graph()
    
    # Add connections
    for source, targets in message_passing.connections.items():
        for target in targets:
            G.add_edge(source, target)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen',
            node_size=1000, arrowsize=20)
    
    path = visualizer._ensure_viz_path(filename)
    plt.savefig(path)
    plt.close()

def plot_belief_state(beliefs: Dict[str, np.ndarray], 
                    uncertainties: Optional[Dict[str, float]] = None,
                    filename: str = None) -> None:
    """Plot belief states and uncertainties.
    
    Args:
        beliefs: Dictionary mapping variable names to belief arrays
        uncertainties: Optional dictionary mapping variable names to uncertainty values
        filename: Path to save the plot
    """
    logger.info("Plotting belief states and uncertainties")
    plt.figure(figsize=(12, 6))
    
    # Create two subplots if we have uncertainties
    if uncertainties is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot beliefs
    variables = list(beliefs.keys())
    values = [np.mean(beliefs[var]) for var in variables]
    
    ax1.bar(variables, values)
    ax1.set_title('Belief States')
    ax1.set_xlabel('Variables')
    ax1.set_ylabel('Mean Belief Value')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot uncertainties if provided
    if uncertainties is not None:
        uncertainty_values = [uncertainties[var] for var in variables]
        ax2.bar(variables, uncertainty_values, color='orange')
        ax2.set_title('Uncertainties')
        ax2.set_xlabel('Variables')
        ax2.set_ylabel('Uncertainty')
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if filename:
        path = visualizer._ensure_viz_path(filename)
        plt.savefig(path)
    plt.close()

def timestamp_filename(prefix: str) -> str:
    """Generate a filename with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.png"

# Export the functions
__all__ = [
    'log_test_case',
    'generate_test_data',
    'plot_thing_hierarchy',
    'plot_free_energy_history',
    'plot_message_network',
    'plot_belief_state',
    'timestamp_filename'
] 