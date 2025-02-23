import os
import logging
import datetime
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure plotting style
sns.set_style('whitegrid')  # Using seaborn's style directly instead of plt.style

def log_test_case(test_name: str, test_data: Dict[str, Any]) -> None:
    """Log test case information."""
    logging.info(f"Running test: {test_name}")
    for key, value in test_data.items():
        logging.debug(f"{key}: {value}")

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

# Export the functions
__all__ = ['log_test_case', 'generate_test_data']

# Remove the self-import
if 'test_utils' in locals():
    del test_utils