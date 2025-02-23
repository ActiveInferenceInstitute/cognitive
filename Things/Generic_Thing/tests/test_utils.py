"""Test utilities and helper functions."""

import logging
import functools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Callable

logger = logging.getLogger(__name__)

def log_test_case(func: Callable) -> Callable:
    """Decorator to log test case execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Running test: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Test completed: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Test failed: {func.__name__}")
            logger.error(f"Error: {str(e)}")
            logger.error("Test state:", exc_info=True)
            raise
    return wrapper

def generate_test_data(size: int = 10) -> Dict[str, np.ndarray]:
    """Generate test data for Generic Thing testing."""
    return {
        'internal_states': np.random.rand(size, 3),
        'external_states': np.random.rand(size, 2),
        'sensory_states': np.random.rand(size, 2),
        'active_states': np.random.rand(size, 2),
        'free_energies': np.random.rand(size),
        'prior': np.random.rand(size, 3),
        'posterior': np.random.rand(size, 3),
        'observations': np.random.rand(size, 2),
        'internal_states_mb': np.random.rand(size, 3),
        'timesteps': np.arange(size)
    }

def plot_message_network(mp: Any, filename: str) -> None:
    """Create visualization of message passing network."""
    G = nx.DiGraph()
    
    # Add nodes and edges
    for source, targets in mp.connections.items():
        G.add_node(source)
        for target in targets:
            G.add_node(target)
            G.add_edge(source, target)
    
    # Create layout
    pos = nx.spring_layout(G)
    
    plt.figure(figsize=(10, 8))
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000, alpha=0.6)
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                          arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos)
    
    plt.title("Message Passing Network")
    plt.axis('off')
    
    # Save visualization
    viz_dir = Path(__file__).parent.parent / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(viz_dir / filename)
    plt.close()
    
    logger.info("Plotting message network")

def plot_thing_hierarchy(root: Any, filename: str) -> None:
    """Create visualization of Thing hierarchy."""
    G = nx.DiGraph()
    
    def add_thing_to_graph(thing: Any) -> None:
        """Recursively add thing and its children to graph."""
        G.add_node(thing.id, name=thing.name)
        for child in thing.children:
            G.add_node(child.id, name=child.name)
            G.add_edge(thing.id, child.id)
            add_thing_to_graph(child)
    
    add_thing_to_graph(root)
    
    # Create layout
    pos = nx.spring_layout(G)
    
    plt.figure(figsize=(12, 8))
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color='lightgreen',
                          node_size=1000, alpha=0.6)
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                          arrows=True, arrowsize=20)
    
    # Add labels with thing names
    labels = {node: G.nodes[node]['name'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels)
    
    plt.title("Thing Hierarchy")
    plt.axis('off')
    
    # Save visualization
    viz_dir = Path(__file__).parent.parent / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(viz_dir / filename)
    plt.close()
    
    logger.info("Plotting thing hierarchy")

def timestamp_filename(base: str) -> str:
    """Generate timestamped filename."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base}_{timestamp}.png"

def plot_free_energy_history(fe_history: list, filename: str) -> None:
    """Create visualization of free energy minimization."""
    plt.figure(figsize=(10, 6))
    plt.plot(fe_history, 'b-', label='Free Energy')
    plt.title('Free Energy Minimization')
    plt.xlabel('Iteration')
    plt.ylabel('Free Energy')
    plt.grid(True)
    plt.legend()
    
    # Save visualization
    viz_dir = Path(__file__).parent.parent / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(viz_dir / filename)
    plt.close()
    
    logger.info("Plotting free energy history")

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path 