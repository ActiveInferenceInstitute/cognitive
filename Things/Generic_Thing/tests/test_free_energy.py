"""Tests for Free Energy functionality."""

import unittest
import logging
import numpy as np
from typing import Dict, Any
from generic_thing.free_energy import FreeEnergy
from generic_thing.tests.test_utils import (
    log_test_case,
    generate_test_data,
    plot_free_energy_history,
    timestamp_filename
)

logger = logging.getLogger(__name__)

class TestFreeEnergy(unittest.TestCase):
    """Test suite for FreeEnergy class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.free_energy = FreeEnergy()
        logger.info("Created FreeEnergy test instance")
        
        # Set up test parameters
        self.model_params = {
            'weights': np.array([[0.5, -0.3], [0.2, 0.4]]),
            'bias': np.array([0.1, -0.1])
        }
        self.recog_params = {
            'mean': np.array([0.0, 0.0]),
            'variance': np.array([1.0, 1.0])
        }
        
        self.free_energy.update_parameters(
            model_params=self.model_params,
            recog_params=self.recog_params
        )
    
    @log_test_case
    def test_initialization(self):
        """Test proper initialization of FreeEnergy."""
        self.assertEqual(len(self.free_energy.history), 0)
        self.assertEqual(self.free_energy.model_parameters, self.model_params)
        self.assertEqual(self.free_energy.recognition_parameters, self.recog_params)
    
    @log_test_case
    def test_free_energy_computation(self):
        """Test computation of variational free energy."""
        # Generate test states and observation
        internal_state = {'x': 0.5, 'y': -0.3}
        external_state = {'x': 0.8, 'y': -0.1}
        observation = {'x': 0.7, 'y': -0.2}
        
        # Compute free energy
        fe = self.free_energy.compute_free_energy(
            internal_state=internal_state,
            external_state=external_state,
            observation=observation
        )
        
        # Verify computation
        self.assertIsInstance(fe, float)
        self.assertGreaterEqual(fe, 0.0)  # Free energy should be non-negative
        self.assertEqual(len(self.free_energy.history), 1)
        
        logger.info(f"Computed free energy: {fe}")
    
    @log_test_case
    def test_free_energy_minimization(self):
        """Test minimization of free energy."""
        # Generate test data
        internal_state = generate_test_data(size=2)
        external_state = generate_test_data(size=2)
        observation = generate_test_data(size=2)
        
        # Minimize free energy
        optimized_state = self.free_energy.minimize(
            internal_state=internal_state,
            external_state=external_state,
            observation=observation
        )
        
        # Verify optimization
        initial_fe = self.free_energy.compute_free_energy(
            internal_state=internal_state,
            external_state=external_state,
            observation=observation
        )
        
        final_fe = self.free_energy.compute_free_energy(
            internal_state=internal_state,
            external_state=optimized_state,
            observation=observation
        )
        
        self.assertLess(final_fe, initial_fe)
        
        # Plot free energy history
        plot_free_energy_history(
            self.free_energy.history,
            timestamp_filename("free_energy_minimization")
        )
    
    @log_test_case
    def test_action_selection(self):
        """Test action selection based on prediction errors."""
        internal_state = {'x': 0.5, 'y': -0.3}
        external_state = {'x': 0.8, 'y': -0.1}
        
        actions = self.free_energy.select_action(
            internal_state=internal_state,
            external_state=external_state
        )
        
        # Verify actions
        self.assertIn('action_x', actions)
        self.assertIn('action_y', actions)
        
        # Verify action directions
        # When external > internal (positive error), action should be negative
        # When external < internal (negative error), action should be positive
        self.assertLess(actions['action_x'], 0)  # external (0.8) > internal (0.5), so negative action
        self.assertLess(actions['action_y'], 0)  # external (-0.1) > internal (-0.3), so negative action
        
        logger.info(f"Selected actions: {actions}")
    
    @log_test_case
    def test_parameter_updates(self):
        """Test updating model and recognition parameters."""
        new_model_params = {
            'weights': np.random.randn(2, 2),
            'bias': np.random.randn(2)
        }
        new_recog_params = {
            'mean': np.random.randn(2),
            'variance': np.abs(np.random.randn(2))
        }
        
        self.free_energy.update_parameters(
            model_params=new_model_params,
            recog_params=new_recog_params
        )
        
        # Verify updates
        for key, value in new_model_params.items():
            self.assertTrue(np.array_equal(
                self.free_energy.model_parameters[key],
                value
            ))
        
        for key, value in new_recog_params.items():
            self.assertTrue(np.array_equal(
                self.free_energy.recognition_parameters[key],
                value
            ))
    
    @log_test_case
    def test_history_tracking(self):
        """Test tracking of free energy history."""
        n_steps = 10
        
        # Generate sequence of states and observations
        for _ in range(n_steps):
            internal_state = generate_test_data(size=2)
            external_state = generate_test_data(size=2)
            observation = generate_test_data(size=2)
            
            self.free_energy.compute_free_energy(
                internal_state=internal_state,
                external_state=external_state,
                observation=observation
            )
        
        # Verify history
        self.assertEqual(len(self.free_energy.history), n_steps)
        self.assertTrue(all(isinstance(fe, float) for fe in self.free_energy.history))
        
        # Plot history
        plot_free_energy_history(
            self.free_energy.history,
            timestamp_filename("free_energy_history")
        )
    
    @log_test_case
    def test_error_handling(self):
        """Test error handling in free energy computations."""
        # Test with incompatible state dimensions
        with self.assertRaises(ValueError):
            self.free_energy.compute_free_energy(
                internal_state={'x': 0.5},  # Missing 'y'
                external_state={'x': 0.8, 'y': -0.1},
                observation={'x': 0.7, 'y': -0.2}
            )
        
        # Test with invalid parameter updates
        with self.assertRaises(ValueError):
            self.free_energy.update_parameters(
                model_params={'invalid': None}
            )
    
    @log_test_case
    def test_expected_free_energy(self):
        """Test computation of expected free energy for action selection."""
        # Set up test states
        internal_state = {'x': 0.5, 'y': -0.3}
        external_state = {'x': 0.8, 'y': -0.1}
        
        # Set up recognition parameters with uncertainties
        self.free_energy.recognition_parameters = {
            'x': np.array([0.5, 0.6]),  # Some uncertainty in x
            'y': np.array([-0.3, -0.2])  # Some uncertainty in y
        }
        
        # Test action that reduces prediction error
        good_action = {
            'action_x': -0.3,  # Moves external state towards internal state
            'action_y': -0.2   # Moves external state towards internal state
        }
        
        # Test action that increases prediction error
        bad_action = {
            'action_x': 0.3,   # Moves external state away from internal state
            'action_y': 0.2    # Moves external state away from internal state
        }
        
        # Compute EFE for both actions
        good_efe = self.free_energy.compute_expected_free_energy(
            internal_state=internal_state,
            external_state=external_state,
            action=good_action
        )
        
        bad_efe = self.free_energy.compute_expected_free_energy(
            internal_state=internal_state,
            external_state=external_state,
            action=bad_action
        )
        
        # Verify that good action has lower EFE
        self.assertLess(good_efe, bad_efe)
        
        # Test information gain component
        # Action that leads to more uncertain state should have lower EFE
        # This encourages exploration of uncertain states
        uncertain_state = {'x': 0.5, 'y': -0.3}
        certain_state = {'x': 0.5, 'y': -0.3}
        
        self.free_energy.recognition_parameters['x'] = np.array([0.5, 0.6, 0.7])  # High uncertainty
        uncertain_efe = self.free_energy.compute_expected_free_energy(
            internal_state=internal_state,
            external_state=uncertain_state,
            action=good_action
        )
        
        self.free_energy.recognition_parameters['x'] = np.array([0.5, 0.51])  # Low uncertainty
        certain_efe = self.free_energy.compute_expected_free_energy(
            internal_state=internal_state,
            external_state=certain_state,
            action=good_action
        )
        
        # Verify that action leading to more uncertain state has lower EFE
        # This encourages exploration of uncertain states
        self.assertLess(uncertain_efe, certain_efe)
        
        logger.info(f"EFE comparison - Good action: {good_efe}, Bad action: {bad_efe}")
        logger.info(f"EFE comparison - Certain: {certain_efe}, Uncertain: {uncertain_efe}")

if __name__ == '__main__':
    unittest.main() 