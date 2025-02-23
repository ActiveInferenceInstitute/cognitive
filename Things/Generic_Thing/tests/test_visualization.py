"""Tests for Visualization functionality."""

import unittest
import logging
import numpy as np
import networkx as nx
from typing import Dict, Any
from generic_thing.visualization import Visualizer
from generic_thing.tests.test_utils import (
    log_test_case,
    generate_test_data,
    timestamp_filename
)
import pytest
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class TestVisualization(unittest.TestCase):
    """Test suite for visualization tools."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.viz = Visualizer()
        logger.info("Created Visualizer test instance")
    
    @log_test_case
    def test_free_energy_landscape(self):
        """Test plotting free energy landscape."""
        # Generate test data
        internal_states = np.linspace(-2, 2, 20)
        external_states = np.linspace(-2, 2, 20)
        
        # Create free energy surface
        X, Y = np.meshgrid(internal_states, external_states)
        Z = (X - Y)**2  # Simple quadratic free energy
        
        # Plot landscape
        save_path = 'test_fe_landscape.png'
        self.viz.plot_free_energy_landscape(
            internal_states=internal_states,
            external_states=external_states,
            free_energies=Z,
            save_path=save_path
        )
        
        # Verify file was created
        self.assertTrue((self.viz.viz_dir / save_path).exists())
    
    @log_test_case
    def test_message_network_dynamics(self):
        """Test plotting message passing network."""
        # Create test network
        connections = {
            'A': ['B', 'C'],
            'B': ['C', 'D'],
            'C': ['D'],
            'D': ['A']
        }
        
        # Generate test messages
        messages = [
            {'source': 'A', 'target': 'B', 'content': {'data': 1}},
            {'source': 'B', 'target': 'C', 'content': {'data': 2}},
            {'source': 'C', 'target': 'D', 'content': {'data': 3}},
            {'source': 'A', 'target': 'B', 'content': {'data': 4}}  # Repeat path
        ]
        
        # Plot network
        filename = 'test_message_network.png'
        self.viz.plot_message_network_dynamics(
            messages=messages,
            connections=connections,
            filename=filename
        )
        
        # Verify file was created
        self.assertTrue((self.viz.viz_dir / filename).exists())
    
    @log_test_case
    def test_markov_blanket_states(self):
        """Test plotting Markov blanket states."""
        # Generate test states
        n_samples = 100
        sensory_states = np.random.randn(n_samples)
        internal_states = np.random.randn(n_samples)
        active_states = np.random.randn(n_samples)
        
        # Plot states
        save_path = 'test_mb_states.png'
        self.viz.plot_markov_blanket_states(
            sensory_states=sensory_states,
            internal_states=internal_states,
            active_states=active_states,
            save_path=save_path
        )
        
        # Verify file was created
        self.assertTrue((self.viz.viz_dir / save_path).exists())
    
    @log_test_case
    def test_belief_updates(self):
        """Test plotting belief updates."""
        # Generate test data
        n_timesteps = 10
        prior = np.random.rand(n_timesteps)
        posterior = np.random.rand(n_timesteps)
        observations = np.random.rand(n_timesteps)
        timesteps = np.arange(n_timesteps)
        
        # Plot belief updates
        save_path = 'test_belief_updates.png'
        self.viz.plot_belief_updates(
            prior=prior,
            posterior=posterior,
            observations=observations,
            timesteps=timesteps,
            save_path=save_path
        )
        
        # Verify file was created
        self.assertTrue((self.viz.viz_dir / save_path).exists())
    
    def tearDown(self):
        """Clean up test files."""
        plt.close('all')  # Close any open figures

@pytest.fixture
def visualizer():
    """Create a Visualizer instance for testing."""
    return Visualizer()

@pytest.fixture
def sample_data():
    """Generate sample data for testing visualizations."""
    internal_states = np.linspace(-2, 2, 20)
    external_states = np.linspace(-2, 2, 20)
    free_energies = np.random.rand(20, 20)
    
    prior = np.random.rand(10)
    posterior = np.random.rand(10)
    observations = np.random.rand(10)
    timesteps = np.arange(10)
    
    sensory_states = np.random.rand(5)
    internal_states_mb = np.random.rand(5)
    active_states = np.random.rand(5)
    
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

def test_plot_free_energy_landscape(visualizer, sample_data):
    """Test plotting of free energy landscape."""
    save_path = 'free_energy_landscape.png'
    
    visualizer.plot_free_energy_landscape(
        internal_states=sample_data['internal_states'],
        external_states=sample_data['external_states'],
        free_energies=sample_data['free_energies'],
        save_path=save_path
    )
    
    assert (visualizer.viz_dir / save_path).exists()
    plt.close()

def test_plot_belief_updates(visualizer, sample_data):
    """Test plotting of belief updates."""
    save_path = 'belief_updates.png'
    
    visualizer.plot_belief_updates(
        prior=sample_data['prior'],
        posterior=sample_data['posterior'],
        observations=sample_data['observations'],
        timesteps=sample_data['timesteps'],
        save_path=save_path
    )
    
    assert (visualizer.viz_dir / save_path).exists()
    plt.close()

def test_plot_markov_blanket_states(visualizer, sample_data):
    """Test plotting of Markov blanket state relationships."""
    save_path = 'markov_blanket.png'
    
    visualizer.plot_markov_blanket_states(
        sensory_states=sample_data['sensory_states'],
        internal_states=sample_data['internal_states_mb'],
        active_states=sample_data['active_states'],
        save_path=save_path
    )
    
    assert (visualizer.viz_dir / save_path).exists()
    plt.close()

if __name__ == '__main__':
    unittest.main() 