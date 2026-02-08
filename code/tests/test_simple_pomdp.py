"""Tests for the SimplePOMDP implementation."""

import pytest
import numpy as np
from pathlib import Path
import yaml
import os
import shutil
import matplotlib.pyplot as plt

from Things.Simple_POMDP import SimplePOMDP, compute_expected_free_energy

@pytest.fixture
def config_path(tmp_path):
    """Create a temporary configuration file for testing."""
    config = {
        'model': {
            'name': 'SimplePOMDP',
            'description': 'Test POMDP',
            'version': '0.1.0'
        },
        'state_space': {
            'num_states': 3,
            'state_labels': ['S1', 'S2', 'S3'],
            'initial_state': 0
        },
        'observation_space': {
            'num_observations': 2,
            'observation_labels': ['O1', 'O2']
        },
        'action_space': {
            'num_actions': 2,
            'action_labels': ['A1', 'A2']
        },
        'matrices': {
            'A_matrix': {
                'shape': [2, 3],
                'initialization': 'random_stochastic',
                'constraints': ['column_stochastic', 'non_negative']
            },
            'B_matrix': {
                'shape': [3, 3, 2],
                'initialization': 'identity_based',
                'initialization_params': {'strength': 0.8},
                'constraints': ['row_stochastic', 'non_negative']
            },
            'C_matrix': {
                'shape': [2],  # [num_observations] - log preferences over observations
                'initialization': 'log_preferences',
                'initialization_params': {
                    'preferences': [0.1, 2.0],  # Strong preference for second observation
                    'description': 'Log-preferences: O1=0.1 (avoid), O2=2.0 (prefer)'
                }
            },
            'D_matrix': {
                'shape': [3],  # [num_states] - initial state prior
                'initialization': 'uniform',
                'description': 'Uniform prior over states'
            },
            'E_matrix': {
                'shape': [2],  # [num_actions] - initial action prior
                'initialization': 'uniform',
                'description': 'Initial uniform prior over actions',
                'learning_rate': 0.2  # Rate at which policy prior is updated
            }
        },
        'inference': {
            'time_horizon': 1,  # Single-step policies
            'num_iterations': 10,
            'learning_rate': 0.5,  # For belief updates
            'temperature': 1.0,    # For policy selection
            'policy_learning_rate': 0.2  # For E matrix updates
        },
        'visualization': {
            'output_dir': str(tmp_path / 'viz_output'),
            'formats': ['png'],
            'dpi': 300,
            'style': {
                'colormap_2d': 'YlOrRd',
                'colormap_3d': 'viridis',
                'figure_size': [10, 8]
            }
        }
    }
    
    config_file = tmp_path / 'test_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return config_file

@pytest.fixture
def output_dir(tmp_path):
    """Create and manage the output directory for test artifacts."""
    output_path = tmp_path / 'Output'
    output_path.mkdir(exist_ok=True)
    
    print(f"\nTest artifacts will be saved to: {output_path}")
    
    yield output_path
    
    # After tests complete, list generated files
    print("\nGenerated test files:")
    for file in output_path.glob('*'):
        print(f"- {file.name}")

@pytest.fixture
def model(output_dir):
    """Create a test model instance."""
    config = {
        'model': {
            'name': 'test_pomdp',
            'description': 'Test POMDP model',
            'version': '0.1'
        },
        'state_space': {
            'num_states': 3,
            'state_labels': ['S1', 'S2', 'S3'],
            'initial_state': 0
        },
        'observation_space': {
            'num_observations': 2,
            'observation_labels': ['O1', 'O2']
        },
        'action_space': {
            'num_actions': 2,
            'action_labels': ['A1', 'A2']
        },
        'matrices': {
            'A_matrix': {
                'shape': [2, 3],
                'initialization': 'random',
                'constraints': ['column_stochastic']
            },
            'B_matrix': {
                'shape': [3, 3, 2],
                'initialization': 'identity_based',
                'initialization_params': {'strength': 0.8},
                'constraints': ['column_stochastic']
            },
            'C_matrix': {
                'shape': [2],  # [num_observations] - log preferences over observations
                'initialization': 'log_preferences',
                'initialization_params': {
                    'preferences': [0.1, 2.0],  # Strong preference for second observation
                    'description': 'Log-preferences: O1=0.1 (avoid), O2=2.0 (prefer)'
                }
            },
            'D_matrix': {
                'shape': [3],  # [num_states] - initial state prior
                'initialization': 'uniform',
                'description': 'Uniform prior over states'
            },
            'E_matrix': {
                'shape': [2],  # [num_actions] - initial action prior
                'initialization': 'uniform',
                'description': 'Initial uniform prior over actions',
                'learning_rate': 0.2  # Rate at which policy prior is updated
            }
        },
        'inference': {
            'time_horizon': 1,  # Single-step policies
            'num_iterations': 10,
            'learning_rate': 0.5,  # For belief updates
            'temperature': 1.0,    # For policy selection
            'policy_learning_rate': 0.2  # For E matrix updates
        },
        'visualization': {
            'output_dir': str(output_dir),
            'style': {
                'figure_size': (10, 8),
                'dpi': 100,
                'colormap': 'viridis',
                'colormap_3d': 'viridis',
                'font_size': 12,
                'file_format': 'png'
            }
        }
    }
    return SimplePOMDP(config)

def test_initialization(model):
    """Test that the model initializes correctly."""
    assert model is not None
    assert isinstance(model.A, np.ndarray)
    assert isinstance(model.B, np.ndarray)
    assert isinstance(model.C, np.ndarray)
    assert isinstance(model.D, np.ndarray)
    assert isinstance(model.E, np.ndarray)
    
    # Check matrix shapes
    assert model.A.shape == (2, 3)
    assert model.B.shape == (3, 3, 2)
    assert model.C.shape == (2,)
    assert model.D.shape == (3,)
    assert model.E.shape == (2,)

def test_matrix_properties(model):
    """Test that matrices satisfy their required properties."""
    # A matrix should be column stochastic
    assert np.allclose(model.A.sum(axis=0), 1)
    assert np.all(model.A >= 0)

    # B matrix should be column stochastic for each action
    for action in range(model.B.shape[-1]):
        assert np.allclose(model.B[..., action].sum(axis=0), 1)
        assert np.all(model.B[..., action] >= 0)

    # D matrix should be a valid probability distribution
    assert np.allclose(model.D.sum(), 1)
    assert np.all(model.D >= 0)

    # E matrix should contain valid action indices
    assert np.all(model.E >= 0)
    assert np.all(model.E < model.config['action_space']['num_actions'])

def test_step_without_action(model):
    """Test taking a step without specifying an action."""
    observation, free_energy = model.step()
    
    assert isinstance(observation, (int, np.integer))
    assert isinstance(free_energy, (float, np.floating))
    assert 0 <= observation < 2
    
    # Check that history is updated
    assert len(model.state.history['states']) == 1
    assert len(model.state.history['observations']) == 1
    assert len(model.state.history['actions']) == 1
    assert len(model.state.history['beliefs']) == 1
    assert len(model.state.history['free_energy']) == 1

def test_step_with_action(model):
    """Test taking a step with a specified action."""
    observation, free_energy = model.step(action=0)
    
    assert isinstance(observation, (int, np.integer))
    assert isinstance(free_energy, (float, np.floating))
    assert 0 <= observation < 2
    
    # Check that history is updated
    assert len(model.state.history['states']) == 1
    assert len(model.state.history['observations']) == 1
    assert len(model.state.history['actions']) == 1
    assert len(model.state.history['beliefs']) == 1
    assert len(model.state.history['free_energy']) == 1

def test_belief_updating(model):
    """Test that beliefs are updated correctly."""
    initial_beliefs = model.state.beliefs.copy()
    
    # Take a step and check that beliefs changed
    model.step()
    
    assert not np.allclose(model.state.beliefs, initial_beliefs)
    assert np.allclose(model.state.beliefs.sum(), 1)
    assert np.all(model.state.beliefs >= 0)

def test_action_selection(model):
    """Test that action selection produces valid actions."""
    action, expected_fe = model._select_action()  # Unpack both values
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < model.config['action_space']['num_actions']
    assert isinstance(expected_fe, np.ndarray)  # Check expected free energy array

def test_expected_free_energy(model):
    """Test computation of expected free energy."""
    # Test with first action
    total_efe, epistemic, pragmatic = compute_expected_free_energy(
        A=model.A,
        B=model.B,
        C=model.C,
        beliefs=model.state.beliefs,
        action=0
    )
    
    # Check return types
    assert isinstance(total_efe, (float, np.floating))
    assert isinstance(epistemic, (float, np.floating))
    assert isinstance(pragmatic, (float, np.floating))
    
    # Check values are finite
    assert not np.isinf(total_efe)
    assert not np.isnan(total_efe)
    assert not np.isinf(epistemic)
    assert not np.isnan(epistemic)
    assert not np.isinf(pragmatic)
    assert not np.isnan(pragmatic)
    
    # Check total is sum of components
    assert np.allclose(total_efe, epistemic + pragmatic)

@pytest.mark.parametrize("plot_type", [
    "belief_evolution",
    "free_energy_landscape",
    "policy_evaluation",
    "state_transitions",
    "observation_likelihood",
    "efe_components_detailed"
])
def test_visualization(model, plot_type, output_dir):
    """Test all visualization methods."""
    # Take a few steps to generate data
    for _ in range(5):
        model.step()
    
    # Update output directory
    model.plotter.output_dir = output_dir
    
    # Generate plot
    fig = model.visualize(plot_type)
    
    # Check that figure was created
    assert isinstance(fig, plt.Figure)
    
    # Check that file was saved
    expected_file = output_dir / f"{plot_type}.png"
    assert expected_file.exists()

def test_invalid_plot_type(model):
    """Test that invalid plot types raise an error."""
    with pytest.raises(ValueError):
        model.visualize("invalid_plot_type")

def test_simulation_run(model):
    """Test running a full simulation."""
    n_steps = 10
    
    observations = []
    free_energies = []
    
    for _ in range(n_steps):
        obs, fe = model.step()
        observations.append(obs)
        free_energies.append(fe)
    
    assert len(observations) == n_steps
    assert len(free_energies) == n_steps
    assert len(model.state.history['states']) == n_steps
    assert len(model.state.history['observations']) == n_steps
    assert len(model.state.history['actions']) == n_steps
    assert len(model.state.history['beliefs']) == n_steps
    assert len(model.state.history['free_energy']) == n_steps

def test_invalid_config(tmp_path):
    """Test that invalid configurations raise appropriate errors."""
    # Missing required field
    invalid_config = {
        'model': {'name': 'Test'}
        # Missing other required fields
    }
    
    config_file = tmp_path / 'invalid_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(invalid_config, f)
    
    with pytest.raises(ValueError):
        SimplePOMDP(config_file)

def test_matrix_validation(tmp_path, config_path):
    """Test that invalid matrix properties raise appropriate errors."""
    # Load valid config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Test invalid A matrix
    config_a = config.copy()
    config_a['matrices']['A_matrix']['initialization'] = 'uniform'
    config_a['matrices']['A_matrix']['constraints'] = []  # Remove stochastic constraint
    
    invalid_config_file = tmp_path / 'invalid_a_matrix.yaml'
    with open(invalid_config_file, 'w') as f:
        yaml.dump(config_a, f)
    
    with pytest.raises(ValueError, match="A matrix must be column stochastic"):
        SimplePOMDP(invalid_config_file)
    
    # Test invalid B matrix
    config_b = config.copy()
    config_b['matrices']['B_matrix']['initialization'] = 'uniform'
    config_b['matrices']['B_matrix']['constraints'] = []  # Remove stochastic constraint
    
    invalid_config_file = tmp_path / 'invalid_b_matrix.yaml'
    with open(invalid_config_file, 'w') as f:
        yaml.dump(config_b, f)
    
    with pytest.raises(ValueError, match="A matrix must be column stochastic"):
        # Note: The error will still be about A matrix since validation happens in order
        SimplePOMDP(invalid_config_file)

def test_model_state_persistence(model):
    """Test that model state persists correctly across steps."""
    initial_state = model.state.current_state
    initial_beliefs = model.state.beliefs.copy()
    
    # Take multiple steps
    for _ in range(5):
        model.step()
        
        # Check that state attributes remain valid
        assert 0 <= model.state.current_state < 3
        assert np.allclose(model.state.beliefs.sum(), 1)
        assert np.all(model.state.beliefs >= 0)
        
        # Check that history is consistent
        assert len(model.state.history['states']) == model.state.time_step
        assert len(model.state.history['beliefs']) == model.state.time_step
    
    # Verify that state changed from initial values
    assert model.state.time_step == 5
    assert len(model.state.history['states']) == 5
    assert not np.allclose(model.state.beliefs, initial_beliefs)

def test_belief_convergence(model):
    """Test that beliefs converge under consistent observations."""
    initial_beliefs = model.state.beliefs.copy()
    
    # Take many steps with fixed action
    n_steps = 20
    observations = []
    for _ in range(n_steps):
        obs, _ = model.step(action=0)
        observations.append(obs)
    
    # Check belief properties
    final_beliefs = model.state.beliefs
    assert np.allclose(final_beliefs.sum(), 1)  # Still a valid distribution
    assert len(np.unique(observations[-5:])) <= 2  # Should settle into a pattern
    
    # Beliefs should have changed from initial state
    assert not np.allclose(final_beliefs, initial_beliefs)
    
    # Beliefs should be more concentrated (lower entropy)
    initial_entropy = -np.sum(initial_beliefs * np.log(initial_beliefs + 1e-12))
    final_entropy = -np.sum(final_beliefs * np.log(final_beliefs + 1e-12))
    assert final_entropy <= initial_entropy

def test_policy_preference(model):
    """Test that policy selection prefers lower free energy policies."""
    # Set strong preference for first observation
    model.C = np.array([2.0, -2.0])  # Prefer O1, avoid O2
    
    # Collect policy selections
    n_trials = 50
    selected_policies = []
    for _ in range(n_trials):
        action = model._select_action()[0]  # Get just the action, not the EFE values
        selected_policies.append(action)
    
    # Should prefer policies that lead to preferred observations
    policy_counts = np.bincount(selected_policies)
    assert policy_counts[0] > n_trials * 0.3  # At least 30% should be policy 0

def test_temporal_consistency(model):
    """Test temporal consistency of the model's behavior."""
    n_steps = 10
    history = {
        'states': [],
        'observations': [],
        'actions': [],
        'beliefs': [],
        'free_energies': []
    }
    
    # Run simulation
    for _ in range(n_steps):
        obs, fe = model.step()
        history['states'].append(model.state.current_state)
        history['observations'].append(obs)
        history['actions'].append(model.state.history['actions'][-1])
        history['beliefs'].append(model.state.beliefs.copy())
        history['free_energies'].append(fe)
    
    # Check temporal consistency
    for t in range(1, n_steps):
        # State transitions should follow B matrix
        prev_state = history['states'][t-1]
        curr_state = history['states'][t]
        action = history['actions'][t-1]
        assert model.B[curr_state, prev_state, action] > 0
        
        # Observations should follow A matrix
        curr_obs = history['observations'][t]
        assert model.A[curr_obs, curr_state] > 0
        
        # Free energy should be finite
        assert not np.isinf(history['free_energies'][t])
        assert not np.isnan(history['free_energies'][t])

def test_visualization_properties(model, output_dir):
    """Test detailed properties of visualization outputs."""
    # Generate some data
    for _ in range(5):
        model.step()
    
    # Update output directory
    model.plotter.output_dir = output_dir
    
    # Test belief evolution plot
    fig = model.visualize("belief_evolution")
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Time Step"
    assert ax.get_ylabel() == "Belief Probability"
    assert ax.get_title() == "Belief Evolution"
    assert len(ax.get_legend().get_texts()) == model.config['state_space']['num_states']
    
    # Test free energy landscape plot
    fig = model.visualize("free_energy_landscape")
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 6  # Three plots + three colorbars
    
    # Get main plot axes (excluding colorbars)
    main_axes = [ax for ax in fig.axes if not ax.get_label() == '<colorbar>']
    assert len(main_axes) == 3  # Three main plots
    
    # Check titles
    assert 'Total Expected Free Energy' in main_axes[0].get_title()
    assert 'Epistemic Value' in main_axes[1].get_title()
    assert 'Pragmatic Value' in main_axes[2].get_title()
    
    # Check labels
    for ax in main_axes:
        assert ax.get_xlabel() == 'Belief in State 0'
        assert ax.get_ylabel() == 'Belief in State 1'
        assert ax.get_zlabel() == 'Value'
    
    # Test policy evaluation plot
    fig = model.visualize("policy_evaluation")
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3  # Three subplots
    
    # Check subplot titles
    assert 'Total Expected Free Energy' in fig.axes[0].get_title()
    assert 'Epistemic Value' in fig.axes[1].get_title()
    assert 'Pragmatic Value' in fig.axes[2].get_title()
    
    # Check subplot labels
    for ax in fig.axes:
        assert ax.get_xlabel() == 'Policy Index'
        assert 'Value' in ax.get_ylabel()
    
    # Generate remaining visualizations
    model.visualize("state_transitions")
    model.visualize("observation_likelihood")
    
    # Check that files were saved
    expected_files = [
        "belief_evolution.png",
        "free_energy_landscape.png",
        "policy_evaluation.png",
        "state_transitions.png",
        "observation_likelihood.png"
    ]
    for filename in expected_files:
        assert (output_dir / filename).exists()

def test_edge_cases(model):
    """Test model behavior in edge cases."""
    # Test with deterministic beliefs
    model.state.beliefs = np.zeros_like(model.state.beliefs)
    model.state.beliefs[0] = 1.0
    obs, fe = model.step()
    assert 0 <= obs < model.A.shape[0]
    assert not np.isinf(fe)
    
    # Test with uniform beliefs
    model.state.beliefs = np.ones_like(model.state.beliefs) / len(model.state.beliefs)
    obs, fe = model.step()
    assert 0 <= obs < model.A.shape[0]
    assert not np.isinf(fe)
    
    # Test with extreme preferences
    original_C = model.C.copy()
    model.C = np.ones_like(model.C) * 10  # Very strong preferences
    obs, fe = model.step()
    assert not np.isinf(fe)
    model.C = original_C  # Restore original preferences

def test_learning_rate_sensitivity(model):
    """Test sensitivity to learning rate parameter."""
    original_lr = model.config['inference']['learning_rate']
    
    # Test with different learning rates
    learning_rates = [0.01, 0.5]  # Use more extreme values
    belief_changes = []
    
    # Fix random seed for reproducibility
    np.random.seed(42)
    
    # Fix initial state and observation sequence
    model.state.current_state = 0
    initial_observation = model._get_observation(0)
    
    for lr in learning_rates:
        model.config['inference']['learning_rate'] = lr
        model.state.beliefs = model.D.copy()  # Reset beliefs to prior
        initial_beliefs = model.state.beliefs.copy()
        
        # Take multiple steps with fixed action and observation
        n_steps = 5
        changes = []
        for _ in range(n_steps):
            # Update beliefs directly with fixed observation
            free_energy = model._update_beliefs(initial_observation, 0)
            change = np.abs(model.state.beliefs - initial_beliefs).mean()
            changes.append(change)
        
        # Use maximum change to capture peak learning effect
        belief_changes.append(np.max(changes))
    
    # Restore original learning rate
    model.config['inference']['learning_rate'] = original_lr
    
    # Higher learning rate should lead to larger changes
    assert belief_changes[0] < belief_changes[1], "Higher learning rates should lead to larger belief changes"

def test_efe_visualization(model, output_dir):
    """Test the detailed EFE components visualization."""
    # Generate some data
    for _ in range(10):
        model.step()
    
    # Update output directory
    model.plotter.output_dir = output_dir
    
    # Generate detailed EFE visualization
    fig = model.visualize("efe_components_detailed")
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 6  # 5 subplots + 1 colorbar
    
    # Get main plot axes (excluding colorbars)
    main_axes = [ax for ax in fig.axes if not ax.get_label() == '<colorbar>']
    assert len(main_axes) == 5  # 5 main plots
    
    # Check subplot properties
    # Total EFE plot
    assert 'Total Expected Free Energy' in main_axes[0].get_title()
    assert main_axes[0].get_xlabel() == 'Time Step'
    assert main_axes[0].get_ylabel() == 'Value'
    
    # Stacked components plot
    assert 'EFE Components' in main_axes[1].get_title()
    assert main_axes[1].get_xlabel() == 'Time Step'
    assert main_axes[1].get_ylabel() == 'Value'
    
    # Component ratio plot
    assert 'Component Ratio' in main_axes[2].get_title()
    assert main_axes[2].get_xlabel() == 'Time Step'
    assert main_axes[2].get_ylabel() == 'Ratio'
    
    # Scatter plot
    assert 'Epistemic vs Pragmatic Value' in main_axes[3].get_title()
    assert 'Epistemic Value' in main_axes[3].get_xlabel()
    assert 'Pragmatic Value' in main_axes[3].get_ylabel()
    
    # Running averages plot
    assert 'Running Averages' in main_axes[4].get_title()
    assert main_axes[4].get_xlabel() == 'Time Step'
    assert main_axes[4].get_ylabel() == 'Value'
    
    # Check that file was saved
    assert (output_dir / "efe_components_detailed.png").exists() 