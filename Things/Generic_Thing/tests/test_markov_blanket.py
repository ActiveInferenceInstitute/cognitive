"""Tests for Markov Blanket functionality."""

import unittest
import logging
import numpy as np
from typing import Dict, Any
from generic_thing.markov_blanket import MarkovBlanket
from generic_thing.tests.test_utils import (
    log_test_case,
    generate_test_data,
    timestamp_filename
)

logger = logging.getLogger(__name__)

class TestMarkovBlanket(unittest.TestCase):
    """Test suite for MarkovBlanket class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.blanket = MarkovBlanket()
        logger.info("Created MarkovBlanket test instance")
        
        # Set up test weights
        self.sensory_weights = np.array([[0.5, -0.3], [0.2, 0.4]])
        self.active_weights = np.array([[0.1, 0.6], [-0.2, 0.3]])
        
        self.blanket.set_weights(
            sensory_weights=self.sensory_weights,
            active_weights=self.active_weights
        )
    
    @log_test_case
    def test_initialization(self):
        """Test proper initialization of MarkovBlanket."""
        self.assertEqual(len(self.blanket.sensory_states), 0)
        self.assertEqual(len(self.blanket.active_states), 0)
        self.assertEqual(len(self.blanket.internal_states), 0)
        self.assertTrue(np.array_equal(self.blanket.sensory_weights, self.sensory_weights))
        self.assertTrue(np.array_equal(self.blanket.active_weights, self.active_weights))
    
    @log_test_case
    def test_state_update(self):
        """Test state updates through observations."""
        # Generate test observation
        observation = {'x': 0.5, 'y': -0.3}
        
        # Update states
        self.blanket.update(observation)
        
        # Verify sensory states updated
        self.assertEqual(self.blanket.sensory_states, observation)
        
        # Verify internal states updated
        self.assertGreater(len(self.blanket.internal_states), 0)
        
        # Verify active states updated
        self.assertGreater(len(self.blanket.active_states), 0)
        
        logger.info(f"Updated states with observation: {observation}")
    
    @log_test_case
    def test_internal_state_computation(self):
        """Test computation of internal states from sensory input."""
        # Set up test sensory states
        self.blanket.sensory_states = {'x': 1.0, 'y': 0.5}
        
        # Update internal states
        self.blanket._update_internal_states()
        
        # Verify computation
        sensory_vec = np.array([1.0, 0.5])
        expected = np.dot(self.sensory_weights, sensory_vec)
        
        for i, value in enumerate(self.blanket.internal_states.values()):
            self.assertAlmostEqual(value, expected[i])
    
    @log_test_case
    def test_active_state_computation(self):
        """Test computation of active states from internal states."""
        # Set up test internal states
        internal_states = {'i1': 0.3, 'i2': -0.2}
        for key, value in internal_states.items():
            self.blanket.internal_states[key] = value
        
        # Update active states
        self.blanket._update_active_states()
        
        # Verify computation
        internal_vec = np.array([0.3, -0.2])
        expected = np.dot(self.active_weights, internal_vec)
        
        for i, value in enumerate(self.blanket.active_states.values()):
            self.assertAlmostEqual(value, expected[i])
    
    @log_test_case
    def test_weight_updates(self):
        """Test updating connection weights."""
        new_sensory_weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        new_active_weights = np.array([[0.5, 0.6], [0.7, 0.8]])
        
        self.blanket.set_weights(
            sensory_weights=new_sensory_weights,
            active_weights=new_active_weights
        )
        
        self.assertTrue(np.array_equal(self.blanket.sensory_weights, new_sensory_weights))
        self.assertTrue(np.array_equal(self.blanket.active_weights, new_active_weights))
    
    @log_test_case
    def test_state_consistency(self):
        """Test consistency between different state representations."""
        observation = generate_test_data(size=2)
        self.blanket.update(observation)
        
        state = self.blanket.get_state()
        
        # Verify all state components present
        self.assertIn('sensory_states', state)
        self.assertIn('active_states', state)
        self.assertIn('internal_states', state)
        
        # Verify sensory states match observation
        for key, value in observation.items():
            self.assertTrue(np.array_equal(
                state['sensory_states'][key],
                value
            ))
    
    @log_test_case
    def test_error_handling(self):
        """Test error handling in state updates and computations."""
        # Set up initial states
        observation = {'x': 0.5, 'y': -0.3}
        self.blanket.update(observation)
        
        # Test with incompatible weight dimensions
        with self.assertRaises(ValueError):
            self.blanket.set_weights(
                sensory_weights=np.array([[1.0]]),  # Wrong shape (should be 2x2)
                active_weights=self.active_weights
            )
        
        # Test with invalid state values
        with self.assertRaises(TypeError):
            self.blanket.update({'invalid': None})

if __name__ == '__main__':
    unittest.main() 