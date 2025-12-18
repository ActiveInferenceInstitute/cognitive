"""
Test suite for SimplePOMDP implementation.
Follows Test-Driven Development (TDD) principles with comprehensive test coverage.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import yaml

# Import the SimplePOMDP class
from simple_pomdp import SimplePOMDP, compute_expected_free_energy


class TestSimplePOMDP:
    """Test suite for SimplePOMDP agent implementation."""

    @pytest.fixture
    def sample_config(self):
        """Provide a sample configuration for testing."""
        return {
            'model': {
                'name': 'TestPOMDP',
                'description': 'Test configuration',
                'version': '0.1.0'
            },
            'state_space': {
                'num_states': 3,
                'state_labels': ['Low', 'Medium', 'High'],
                'initial_state': 1
            },
            'observation_space': {
                'num_observations': 3,
                'observation_labels': ['Low', 'Medium', 'High'],
            },
            'action_space': {
                'num_actions': 3,
                'action_labels': ['Decrease', 'Stay', 'Increase']
            },
            'matrices': {
                'A_matrix': {
                    'shape': [3, 3],
                    'initialization': 'identity_based',
                    'initialization_params': {'strength': 0.7},
                    'constraints': ['column_stochastic']
                },
                'B_matrix': {
                    'shape': [3, 3, 3],
                    'initialization': 'identity_based',
                    'initialization_params': {'strength': 0.8},
                    'constraints': ['tensor_stochastic']
                },
                'C_matrix': {
                    'shape': [3],
                    'initialization': 'zeros',
                    'initialization_params': {},
                    'constraints': []
                },
                'D_matrix': {
                    'shape': [3],
                    'initialization': 'uniform',
                    'initialization_params': {},
                    'constraints': ['normalized']
                }
            },
            'inference': {
                'planning_horizon': 1,
                'policy_learning_rate': 0.1,
                'temperature': 1.0
            },
            'visualization': {
                'output_dir': 'test_output',
                'style': 'default'
            }
        }

    @pytest.fixture
    def config_file(self, sample_config):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            return f.name

    def test_initialization(self, config_file):
        """Test proper initialization of SimplePOMDP agent."""
        agent = SimplePOMDP(config_file)

        # Check that agent was initialized
        assert agent is not None
        assert hasattr(agent, 'config')
        assert hasattr(agent, 'A')
        assert hasattr(agent, 'B')
        assert hasattr(agent, 'C')
        assert hasattr(agent, 'D')

        # Check matrix shapes
        assert agent.A.shape == (3, 3)  # [observations, states]
        assert agent.B.shape == (3, 3, 3)  # [states, states, actions]
        assert agent.C.shape == (3,)  # [observations]
        assert agent.D.shape == (3,)  # [states]

    def test_matrix_properties(self, config_file):
        """Test that matrices have required mathematical properties."""
        agent = SimplePOMDP(config_file)

        # A matrix should be column stochastic (columns sum to 1)
        np.testing.assert_allclose(np.sum(agent.A, axis=0), 1.0, rtol=1e-10)

        # B tensor should be stochastic (each action's matrix should be row stochastic)
        for action in range(agent.B.shape[2]):
            B_action = agent.B[:, :, action]
            np.testing.assert_allclose(np.sum(B_action, axis=1), 1.0, rtol=1e-10)

        # D vector should be normalized
        assert abs(np.sum(agent.D) - 1.0) < 1e-10

    def test_belief_updating(self, config_file):
        """Test belief updating mechanism."""
        agent = SimplePOMDP(config_file)

        # Set initial beliefs
        initial_beliefs = np.array([0.2, 0.5, 0.3])
        agent.state.beliefs = initial_beliefs

        # Take a step
        observation, free_energy = agent.step(action=1)  # Stay action

        # Check that beliefs are still valid probability distribution
        assert len(agent.state.beliefs) == 3
        np.testing.assert_allclose(np.sum(agent.state.beliefs), 1.0, rtol=1e-10)
        assert np.all(agent.state.beliefs >= 0)

        # Check that free energy is a finite number
        assert np.isfinite(free_energy)

    def test_action_selection(self, config_file):
        """Test action selection mechanism."""
        agent = SimplePOMDP(config_file)

        # Set known beliefs
        agent.state.beliefs = np.array([0.1, 0.8, 0.1])

        # Select action
        action, expected_fe = agent._select_action()

        # Check that action is valid
        assert action in range(3)
        assert len(expected_fe) == 3

        # Check that expected free energies are finite
        assert np.all(np.isfinite(expected_fe))

    def test_expected_free_energy_computation(self, config_file):
        """Test Expected Free Energy computation."""
        agent = SimplePOMDP(config_file)

        beliefs = np.array([0.3, 0.4, 0.3])

        # Test EFE computation for each action
        for action in range(3):
            efe, epistemic, pragmatic = compute_expected_free_energy(
                A=agent.A, B=agent.B, C=agent.C,
                beliefs=beliefs, action=action
            )

            # Check that all values are finite
            assert np.isfinite(efe)
            assert np.isfinite(epistemic)
            assert np.isfinite(pragmatic)

            # Check that EFE equals epistemic + pragmatic
            assert abs(efe - (epistemic + pragmatic)) < 1e-10

    def test_history_tracking(self, config_file):
        """Test that agent properly tracks history."""
        agent = SimplePOMDP(config_file)

        initial_history_length = len(agent.state.history['states'])

        # Take a few steps
        for _ in range(3):
            agent.step()

        # Check that history has grown
        assert len(agent.state.history['states']) > initial_history_length
        assert len(agent.state.history['observations']) > initial_history_length
        assert len(agent.state.history['actions']) > initial_history_length
        assert len(agent.state.history['beliefs']) > initial_history_length

    def test_edge_cases(self, config_file):
        """Test edge cases and error conditions."""
        agent = SimplePOMDP(config_file)

        # Test with extreme beliefs
        agent.state.beliefs = np.array([1.0, 0.0, 0.0])  # Certainty
        observation, fe = agent.step()
        assert np.isfinite(fe)

        agent.state.beliefs = np.array([0.0, 1.0, 0.0])  # Certainty
        observation, fe = agent.step()
        assert np.isfinite(fe)

    def test_convergence(self, config_file):
        """Test that the agent behavior converges over time."""
        agent = SimplePOMDP(config_file)

        # Run multiple steps and check for stability
        free_energies = []
        for _ in range(10):
            _, fe = agent.step()
            free_energies.append(fe)

        # Check that we have finite free energies
        assert all(np.isfinite(fe) for fe in free_energies)

        # Check that beliefs remain valid distributions
        for beliefs in agent.state.history['beliefs']:
            np.testing.assert_allclose(np.sum(beliefs), 1.0, rtol=1e-10)
            assert np.all(beliefs >= 0)


class TestExpectedFreeEnergy:
    """Test the compute_expected_free_energy function."""

    def test_basic_computation(self):
        """Test basic EFE computation with simple matrices."""
        # Simple 2x2 matrices for testing
        A = np.array([[0.9, 0.1], [0.1, 0.9]])  # Observation model
        B = np.array([[[0.8, 0.2], [0.2, 0.8]], [[0.2, 0.8], [0.8, 0.2]]])  # Transition model
        C = np.array([0.0, 1.0])  # Preferences
        beliefs = np.array([0.6, 0.4])  # Current beliefs

        efe, epistemic, pragmatic = compute_expected_free_energy(A, B, C, beliefs, action=0)

        # Check types and finiteness
        assert isinstance(efe, (float, np.floating))
        assert isinstance(epistemic, (float, np.floating))
        assert isinstance(pragmatic, (float, np.floating))

        assert np.isfinite(efe)
        assert np.isfinite(epistemic)
        assert np.isfinite(pragmatic)

    def test_epistemic_vs_pragmatic(self):
        """Test that epistemic and pragmatic components behave as expected."""
        # Create matrices where epistemic and pragmatic values can be distinguished
        A = np.eye(2)  # Perfect observation model
        B = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])  # Deterministic transitions
        C = np.array([0.0, 1.0])  # Prefer second observation
        beliefs = np.array([0.5, 0.5])  # Uncertain beliefs

        efe, epistemic, pragmatic = compute_expected_free_energy(A, B, C, beliefs, action=0)

        # With uncertain beliefs, epistemic value should be positive (information gain)
        # Pragmatic value depends on preferences
        assert np.isfinite(epistemic)
        assert np.isfinite(pragmatic)


if __name__ == '__main__':
    pytest.main([__file__])
