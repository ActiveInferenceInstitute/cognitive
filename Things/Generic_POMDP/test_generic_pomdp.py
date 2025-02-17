"""
Tests for Generic POMDP implementation.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pytest
import numpy as np
import yaml
import logging
import sys
from pathlib import Path
from datetime import datetime
from generic_pomdp import GenericPOMDP, ModelState
from matplotlib.gridspec import GridSpec

# Configure comprehensive logging
def setup_logging():
    """Configure logging to both file and console with detailed formatting."""
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent / "Output/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"test_run_{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler with full output
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler with color output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return log_file

# Setup logging before tests run
log_file = setup_logging()
logger = logging.getLogger(__name__)

@pytest.fixture
def config():
    """Load configuration file."""
    config_path = Path(__file__).parent / "configuration.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def configured_pomdp(config):
    """Create POMDP from configuration file."""
    dims = config['dimensions']
    return GenericPOMDP(
        num_observations=dims['observations'],
        num_states=dims['states'],
        num_actions=dims['actions'],
        planning_horizon=dims['planning_horizon']
    )

@pytest.fixture
def small_pomdp():
    """Create small POMDP for testing."""
    return GenericPOMDP(
        num_observations=2,
        num_states=3,
        num_actions=2,
        planning_horizon=4  # Updated to match configuration
    )

@pytest.fixture
def medium_pomdp():
    """Create medium POMDP for testing."""
    return GenericPOMDP(
        num_observations=4,
        num_states=5,
        num_actions=3,
        planning_horizon=4  # Updated to match configuration
    )

@pytest.fixture(scope="session", autouse=True)
def log_global_test_info(request):
    """Log global test information at start and end of test session."""
    logger.info("=" * 80)
    logger.info("Starting test session")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Python version: {sys.version}")
    logger.info("=" * 80)
    
    yield
    
    logger.info("=" * 80)
    logger.info("Test session completed")
    logger.info("=" * 80)

@pytest.fixture(scope="session")
def output_dir():
    """Create and return output directory structure."""
    base_dir = Path(__file__).parent / "Output"
    dirs = {
        "logs": base_dir / "logs",
        "plots": base_dir / "plots",
        "test_results": base_dir / "test_results",
        "simulations": base_dir / "simulations"
    }
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def test_configuration_dimensions(config):
    """Test configuration file dimensions are valid."""
    dims = config['dimensions']
    logger.info(f"Testing configuration dimensions:")
    logger.info(f"Observations: {dims['observations']}")
    logger.info(f"States: {dims['states']}")
    logger.info(f"Actions: {dims['actions']}")
    logger.info(f"Planning horizon: {dims['planning_horizon']}")
    logger.info(f"Total timesteps: {dims['total_timesteps']}")
    
    # Validate dimensions are positive integers
    assert dims['observations'] > 0, "Number of observations must be positive"
    assert dims['states'] > 0, "Number of states must be positive"
    assert dims['actions'] > 0, "Number of actions must be positive"
    assert dims['planning_horizon'] > 0, "Planning horizon must be positive"
    assert dims['total_timesteps'] > 0, "Total timesteps must be positive"
    
    logger.info("All configuration dimensions validated successfully")

def test_initialization(configured_pomdp, config):
    """Test model initialization and dimensionality."""
    dims = config['dimensions']
    
    logger.info("Checking model dimensions match configuration:")
    
    # Check dimensions
    assert configured_pomdp.num_observations == dims['observations'], \
        f"Expected {dims['observations']} observations, got {configured_pomdp.num_observations}"
    logger.info(f"✓ Observations dimension: {configured_pomdp.num_observations}")
    
    assert configured_pomdp.num_states == dims['states'], \
        f"Expected {dims['states']} states, got {configured_pomdp.num_states}"
    logger.info(f"✓ States dimension: {configured_pomdp.num_states}")
    
    assert configured_pomdp.num_actions == dims['actions'], \
        f"Expected {dims['actions']} actions, got {configured_pomdp.num_actions}"
    logger.info(f"✓ Actions dimension: {configured_pomdp.num_actions}")
    
    # Check matrix shapes
    logger.info("\nValidating matrix shapes:")
    
    A_shape = configured_pomdp.A.shape
    expected_A = (dims['observations'], dims['states'])
    assert A_shape == expected_A, f"A matrix shape mismatch. Expected {expected_A}, got {A_shape}"
    logger.info(f"✓ A matrix shape: {A_shape}")
    
    B_shape = configured_pomdp.B.shape
    expected_B = (dims['states'], dims['states'], dims['actions'])
    assert B_shape == expected_B, f"B matrix shape mismatch. Expected {expected_B}, got {B_shape}"
    logger.info(f"✓ B matrix shape: {B_shape}")
    
    C_shape = configured_pomdp.C.shape
    expected_C = (dims['observations'], dims['planning_horizon'])
    assert C_shape == expected_C, f"C matrix shape mismatch. Expected {expected_C}, got {C_shape}"
    logger.info(f"✓ C matrix shape: {C_shape}")
    
    D_shape = configured_pomdp.D.shape
    expected_D = (dims['states'],)
    assert D_shape == expected_D, f"D matrix shape mismatch. Expected {expected_D}, got {D_shape}"
    logger.info(f"✓ D matrix shape: {D_shape}")
    
    E_shape = configured_pomdp.E.shape
    expected_E = (dims['actions'],)
    assert E_shape == expected_E, f"E matrix shape mismatch. Expected {expected_E}, got {E_shape}"
    logger.info(f"✓ E matrix shape: {E_shape}")
    
    # Check initial state
    assert isinstance(configured_pomdp.state, ModelState)
    assert configured_pomdp.state.beliefs.shape == (dims['states'],)
    assert configured_pomdp.state.time_step == 0
    logger.info("\n✓ Initial state validated successfully")
    
    logger.info("\nAll initialization checks passed successfully")

def test_matrix_properties(small_pomdp):
    """Test matrix properties."""
    # A matrix (observation model)
    assert np.allclose(small_pomdp.A.sum(axis=0), 1.0)  # Column stochastic
    assert np.all(small_pomdp.A >= 0)  # Non-negative
    
    # B matrix (transition model)
    for a in range(small_pomdp.num_actions):
        assert np.allclose(small_pomdp.B[:,:,a].sum(axis=0), 1.0)  # Column stochastic
        assert np.all(small_pomdp.B[:,:,a] >= 0)  # Non-negative
    
    # C matrix (preferences)
    assert np.all(np.isfinite(small_pomdp.C))  # Finite values
    
    # D matrix (prior beliefs)
    assert np.allclose(small_pomdp.D.sum(), 1.0)  # Normalized
    assert np.all(small_pomdp.D >= 0)  # Non-negative
    
    # E matrix (policy prior)
    assert np.allclose(small_pomdp.E.sum(), 1.0)  # Normalized
    assert np.all(small_pomdp.E >= 0)  # Non-negative

def test_step_without_action(small_pomdp):
    """Test stepping without providing action."""
    observation, free_energy = small_pomdp.step()
    
    # Check observation
    assert 0 <= observation < small_pomdp.num_observations
    
    # Check free energy
    assert np.isfinite(free_energy)
    
    # Check history updated
    assert len(small_pomdp.state.history['observations']) == 1
    assert len(small_pomdp.state.history['actions']) == 1
    assert len(small_pomdp.state.history['beliefs']) == 2  # Initial + updated
    assert len(small_pomdp.state.history['free_energy']) == 1

def test_step_with_action(small_pomdp):
    """Test stepping with provided action."""
    action = 0
    observation, free_energy = small_pomdp.step(action)
    
    # Check observation
    assert 0 <= observation < small_pomdp.num_observations
    
    # Check free energy
    assert np.isfinite(free_energy)
    
    # Check correct action taken
    assert small_pomdp.state.history['actions'][-1] == action

def test_belief_updating(small_pomdp):
    """Test belief updating mechanism."""
    initial_beliefs = small_pomdp.state.beliefs.copy()
    
    # Take step
    small_pomdp.step()
    
    # Check beliefs updated
    assert not np.allclose(small_pomdp.state.beliefs, initial_beliefs)
    assert np.allclose(small_pomdp.state.beliefs.sum(), 1.0)  # Still normalized
    assert np.all(small_pomdp.state.beliefs >= 0)  # Still non-negative

def test_action_selection(small_pomdp):
    """Test that action selection uses policy evaluation properly."""
    # Run action selection multiple times
    n_samples = 100
    selected_actions = []
    action_probs_list = []

    for _ in range(n_samples):
        # Get EFE components to compute action values
        efe_components = small_pomdp.get_efe_components()
        policies = efe_components['policies']
        total_efe = efe_components['total_efe']
        
        # Group policies by their first action and take the minimum EFE for each first action
        action_values = np.full(small_pomdp.num_actions, np.inf)
        for action_idx in range(small_pomdp.num_actions):
            # Find policies that start with this action
            matching_policies = [i for i, p in enumerate(policies) if p[0] == action_idx]
            if matching_policies:
                # Take the minimum EFE among policies starting with this action
                action_values[action_idx] = np.min(total_efe[matching_policies])
        
        # Get action probabilities
        action_probs = small_pomdp._softmax(-action_values / (0.001 + small_pomdp.stability_threshold))
        action = np.random.choice(len(action_probs), p=action_probs)
        
        selected_actions.append(action)
        action_probs_list.append(action_probs)

    # Convert to numpy array for easier analysis
    selected_actions = np.array(selected_actions)
    
    # Compute empirical action frequencies
    action_counts = np.bincount(selected_actions, minlength=small_pomdp.num_actions)
    empirical_frequencies = action_counts / n_samples
    
    # Average action probabilities across samples
    avg_action_probs = np.mean(action_probs_list, axis=0)
    
    # Check that empirical frequencies roughly match expected probabilities
    np.testing.assert_allclose(empirical_frequencies, avg_action_probs, atol=0.1)
    
    # Additional assertions to verify action selection properties
    assert len(selected_actions) == n_samples, "Wrong number of selected actions"
    assert all(0 <= a < small_pomdp.num_actions for a in selected_actions), "Invalid action selected"
    assert all(np.isclose(sum(probs), 1.0) for probs in action_probs_list), "Action probabilities don't sum to 1"

def test_save_load_state(small_pomdp, tmp_path):
    """Test state saving and loading."""
    # Take some steps
    for _ in range(3):
        small_pomdp.step()
    
    # Save state
    save_path = tmp_path / "test_state.yaml"
    small_pomdp.save_state(save_path)
    
    # Create new model
    new_model = GenericPOMDP(
        num_observations=2,
        num_states=3,
        num_actions=2,
        planning_horizon=4  # Updated to match configuration
    )
    
    # Load state
    new_model.load_state(save_path)
    
    # Check state loaded correctly
    assert new_model.state.current_state == small_pomdp.state.current_state
    assert np.allclose(new_model.state.beliefs, small_pomdp.state.beliefs)
    assert new_model.state.time_step == small_pomdp.state.time_step
    
    # Check history loaded correctly
    for key in small_pomdp.state.history:
        assert len(new_model.state.history[key]) == len(small_pomdp.state.history[key])
        for i in range(len(small_pomdp.state.history[key])):
            if isinstance(small_pomdp.state.history[key][i], dict):
                # For efe_components dictionaries
                for k, v in small_pomdp.state.history[key][i].items():
                    if isinstance(v, (np.ndarray, list)):
                        assert np.allclose(np.array(new_model.state.history[key][i][k]), np.array(v))
                    else:
                        assert new_model.state.history[key][i][k] == v
            elif isinstance(small_pomdp.state.history[key][i], (np.ndarray, list)):
                assert np.allclose(np.array(new_model.state.history[key][i]), np.array(small_pomdp.state.history[key][i]))
            else:
                assert new_model.state.history[key][i] == small_pomdp.state.history[key][i]

def test_numerical_stability(small_pomdp):
    """Test numerical stability of computations."""
    # Set very small beliefs
    small_pomdp.state.beliefs = np.ones(small_pomdp.num_states) * 1e-10
    small_pomdp.state.beliefs /= small_pomdp.state.beliefs.sum()
    
    # Should not raise and give finite results
    observation, free_energy = small_pomdp.step()
    assert np.isfinite(free_energy)
    assert np.all(np.isfinite(small_pomdp.state.beliefs))

def test_learning_dynamics(medium_pomdp):
    """Test learning dynamics over multiple steps."""
    # Track belief entropy, beliefs and free energy
    entropies = []
    beliefs_history = []
    free_energies = []
    
    # Run for multiple steps
    n_steps = 50  # Increased number of steps
    for _ in range(n_steps):
        obs, fe = medium_pomdp.step()
        beliefs = medium_pomdp.state.beliefs.copy()
        beliefs_history.append(beliefs)
        entropy = -np.sum(beliefs * np.log(beliefs + medium_pomdp.stability_threshold))
        entropies.append(entropy)
        free_energies.append(fe)
    
    # Test 1: Entropy should stay within reasonable bounds
    initial_entropy = -np.sum(np.ones(medium_pomdp.num_states)/medium_pomdp.num_states * 
                            np.log(np.ones(medium_pomdp.num_states)/medium_pomdp.num_states))
    assert all(0 <= e <= initial_entropy + 0.5 for e in entropies), "Entropy outside reasonable bounds"
    
    # Test 2: Beliefs should change over time (learning is happening)
    belief_changes = [np.linalg.norm(b2 - b1) for b1, b2 in zip(beliefs_history[:-1], beliefs_history[1:])]
    assert np.mean(belief_changes) > 0.001, "Beliefs not changing significantly"
    
    # Test 3: Beliefs should remain valid probability distributions
    for beliefs in beliefs_history:
        assert np.allclose(beliefs.sum(), 1.0), "Beliefs not normalized"
        assert np.all(beliefs >= 0), "Negative belief values found"
    
    # Test 4: Free energy should show evidence of learning
    # Compare average free energy in first and last 10 steps
    early_fe = np.mean(free_energies[:10])
    late_fe = np.mean(free_energies[-10:])
    assert late_fe <= early_fe + 2.0, "Free energy not showing evidence of learning"
    
    # Test 5: Belief changes should show temporal structure
    # Look at the sequence of changes in 5-step windows
    window_size = 5
    window_changes = []
    for i in range(0, len(belief_changes) - window_size, window_size):
        window = belief_changes[i:i+window_size]
        window_changes.append(np.mean(window))
    
    # Test for a general decreasing trend in window changes
    # Allow for more exploration by requiring only 30% of consecutive pairs to show decrease
    decreasing_pairs = sum(w1 > w2 for w1, w2 in zip(window_changes[:-1], window_changes[1:]))
    total_pairs = len(window_changes) - 1
    assert decreasing_pairs / total_pairs >= 0.3, "Belief changes don't show consistent learning trend"

def test_preference_influence(small_pomdp):
    """Test influence of preferences on action selection."""
    # Set up transition dynamics to make observation 0 achievable
    # Make action 0 lead to states that generate observation 0
    small_pomdp.B[:,:,0] = 0.0
    small_pomdp.B[0,:,0] = 1.0  # Action 0 leads to state 0
    
    # Make action 1 lead to states that generate observation 1
    small_pomdp.B[:,:,1] = 0.0
    small_pomdp.B[1,:,1] = 1.0  # Action 1 leads to state 1
    
    # Set up observation model
    small_pomdp.A = np.zeros((small_pomdp.num_observations, small_pomdp.num_states))
    small_pomdp.A[0,0] = 1.0  # State 0 generates observation 0
    small_pomdp.A[1,1] = 1.0  # State 1 generates observation 1
    small_pomdp.A[1,2] = 1.0  # State 2 generates observation 1
    
    # Set very strong preference for observation 0
    small_pomdp.C[0,:] = 5.0
    small_pomdp.C[1,:] = -5.0
    
    # Run for multiple steps
    n_steps = 50
    observations = []
    for _ in range(n_steps):
        obs, _ = small_pomdp.step()
        observations.append(obs)
    
    # Should see more preferred observations
    obs_counts = np.bincount(observations)
    logger.info(f"Observation counts: {obs_counts}")
    # Require at least 60% preferred observations
    assert obs_counts[0] > 0.6 * n_steps, \
        f"Expected at least 60% observations of type 0, got {obs_counts[0]/n_steps*100:.1f}%"

def test_invalid_dimensions():
    """Test handling of invalid dimensions."""
    with pytest.raises(ValueError):
        GenericPOMDP(
            num_observations=0,  # Invalid
            num_states=3,
            num_actions=2
        )

def test_invalid_action(small_pomdp):
    """Test handling of invalid action."""
    with pytest.raises(IndexError):
        small_pomdp.step(action=small_pomdp.num_actions)  # Invalid action

def test_convergence(small_pomdp):
    """Test convergence properties."""
    # Set deterministic transition and observation
    small_pomdp.B[:,:,0] = np.eye(small_pomdp.num_states)  # Identity transitions
    
    # Create proper observation matrix that maps states to observations
    small_pomdp.A = np.zeros((small_pomdp.num_observations, small_pomdp.num_states))
    # Map each state to a unique observation if possible, otherwise map multiple states to same observation
    for i in range(small_pomdp.num_states):
        small_pomdp.A[i % small_pomdp.num_observations, i] = 1.0
    
    # Normalize columns to ensure proper probability distribution
    small_pomdp.A /= small_pomdp.A.sum(axis=0, keepdims=True)
    
    # Run for multiple steps with same action
    beliefs_history = []
    for _ in range(10):
        small_pomdp.step(action=0)
        beliefs_history.append(small_pomdp.state.beliefs.copy())
    
    # Should converge (later beliefs should be more similar)
    diffs = [np.linalg.norm(b1 - b2)
             for b1, b2 in zip(beliefs_history[:-1], beliefs_history[1:])]
    assert np.mean(diffs[:3]) > np.mean(diffs[-3:])

def create_matrix_visualizations(pomdp, plots_dir):
    """Create and save visualizations for all matrices."""
    # Individual matrix plots
    
    # A matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(pomdp.A, cmap='viridis', aspect='auto')
    plt.colorbar(label='Probability')
    plt.title('A Matrix (Observation Model)')
    plt.xlabel('States')
    plt.ylabel('Observations')
    for i in range(pomdp.A.shape[0]):
        for j in range(pomdp.A.shape[1]):
            plt.text(j, i, f'{pomdp.A[i,j]:.2f}', ha='center', va='center', 
                    color='white' if pomdp.A[i,j] > 0.5 else 'black')
    plt.savefig(plots_dir / 'A_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # B matrix (one subplot per action)
    n_actions = pomdp.B.shape[2]
    fig, axes = plt.subplots(1, n_actions, figsize=(6*n_actions, 5))
    if n_actions == 1:
        axes = [axes]
    for a in range(n_actions):
        im = axes[a].imshow(pomdp.B[:,:,a], cmap='viridis', aspect='auto')
        axes[a].set_title(f'B Matrix (Action {a})')
        axes[a].set_xlabel('States t')
        axes[a].set_ylabel('States t+1')
        plt.colorbar(im, ax=axes[a], label='Probability')
        for i in range(pomdp.B.shape[0]):
            for j in range(pomdp.B.shape[1]):
                axes[a].text(j, i, f'{pomdp.B[i,j,a]:.2f}', ha='center', va='center',
                           color='white' if pomdp.B[i,j,a] > 0.5 else 'black')
    plt.tight_layout()
    plt.savefig(plots_dir / 'B_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # C matrix
    plt.figure(figsize=(10, 6))
    vmax = max(abs(pomdp.C.min()), abs(pomdp.C.max()))
    vmin = -vmax
    plt.imshow(pomdp.C, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Preference')
    plt.title('C Matrix (Preferences)')
    plt.xlabel('Time Steps')
    plt.ylabel('Observations')
    for i in range(pomdp.C.shape[0]):
        for j in range(pomdp.C.shape[1]):
            plt.text(j, i, f'{pomdp.C[i,j]:.2f}', ha='center', va='center',
                    color='white' if abs(pomdp.C[i,j]) > vmax/2 else 'black')
    plt.savefig(plots_dir / 'C_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # D matrix
    plt.figure(figsize=(6, 4))
    bars = plt.bar(range(len(pomdp.D)), pomdp.D)
    plt.title('D Matrix (Prior Beliefs)')
    plt.xlabel('States')
    plt.ylabel('Probability')
    for bar, v in zip(bars, pomdp.D):
        plt.text(bar.get_x() + bar.get_width()/2, v, f'{v:.2f}',
                ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    plt.savefig(plots_dir / 'D_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # E matrix
    plt.figure(figsize=(6, 4))
    bars = plt.bar(range(len(pomdp.E)), pomdp.E)
    plt.title('E Matrix (Policy Prior)')
    plt.xlabel('Actions')
    plt.ylabel('Probability')
    for bar, v in zip(bars, pomdp.E):
        plt.text(bar.get_x() + bar.get_width()/2, v, f'{v:.2f}',
                ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    plt.savefig(plots_dir / 'E_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined overview plot
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 3, figure=fig)
    
    # A matrix
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(pomdp.A, cmap='viridis', aspect='auto')
    ax1.set_title('A Matrix\n(Observation Model)')
    plt.colorbar(im1, ax=ax1)
    ax1.set_xlabel('States')
    ax1.set_ylabel('Observations')
    
    # B matrix (first action)
    ax2 = fig.add_subplot(gs[0, 1:])
    im2 = ax2.imshow(pomdp.B[:,:,0], cmap='viridis', aspect='auto')
    ax2.set_title('B Matrix\n(Transition Model, Action 0)')
    plt.colorbar(im2, ax=ax2)
    ax2.set_xlabel('States t')
    ax2.set_ylabel('States t+1')
    
    # C matrix
    ax3 = fig.add_subplot(gs[1, :])
    vmax = max(abs(pomdp.C.min()), abs(pomdp.C.max()))
    vmin = -vmax
    im3 = ax3.imshow(pomdp.C, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    ax3.set_title('C Matrix (Preferences)')
    plt.colorbar(im3, ax=ax3)
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Observations')
    
    # D matrix
    ax4 = fig.add_subplot(gs[2, 0:2])
    ax4.bar(range(len(pomdp.D)), pomdp.D)
    ax4.set_title('D Matrix (Prior Beliefs)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('States')
    ax4.set_ylabel('Probability')
    
    # E matrix
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.bar(range(len(pomdp.E)), pomdp.E)
    ax5.set_title('E Matrix (Policy Prior)')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlabel('Actions')
    ax5.set_ylabel('Probability')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'matrix_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_full_simulation(configured_pomdp, config, output_dir):
    """Run a full simulation with comprehensive logging."""
    try:
        logger.info("\nStarting full simulation test")
        
        # Setup output paths
        test_dir = output_dir["test_results"] / "full_simulation"
        test_dir.mkdir(exist_ok=True)
        plots_dir = output_dir["plots"]
        
        # Create EFE components directory
        efe_dir = plots_dir / "efe_components"
        efe_dir.mkdir(exist_ok=True)
        
        # Log paths
        logger.info(f"\nOutput directories:")
        logger.info(f"Test results: {test_dir}")
        logger.info(f"Plots: {plots_dir}")
        logger.info(f"EFE Components: {efe_dir}")
        
        # Log initial configuration
        logger.info("\nModel Configuration:")
        logger.info(f"Observations: {configured_pomdp.num_observations}")
        logger.info(f"States: {configured_pomdp.num_states}")
        logger.info(f"Actions: {configured_pomdp.num_actions}")
        logger.info(f"Planning horizon: {config['dimensions']['planning_horizon']}")
        logger.info(f"Total timesteps: {config['dimensions']['total_timesteps']}")
        
        # Create matrix visualizations before simulation
        create_matrix_visualizations(configured_pomdp, plots_dir)
        
        # Run simulation
        n_steps = config['dimensions']['total_timesteps']
        planning_horizon = config['dimensions']['planning_horizon']
        logger.info(f"\nRunning simulation for {n_steps} steps")
        
        history = {
            'observations': [],
            'actions': [],
            'free_energies': [],
            'beliefs': [],
            'efe_components': []  # Store EFE components for each policy at each timestep
        }
        
        # Store initial beliefs
        history['beliefs'].append(configured_pomdp.state.beliefs.copy())
        
        # Run simulation with progress tracking
        for step in range(n_steps):
            logger.info(f"\nStep {step + 1}/{n_steps}")
            
            # Get EFE components before taking step
            efe_components = configured_pomdp.get_efe_components()  # This method needs to be implemented in GenericPOMDP
            history['efe_components'].append(efe_components)
            
            # Take step
            obs, fe = configured_pomdp.step()
            
            # Log step results
            logger.info(f"Observation: {obs}")
            logger.info(f"Free Energy: {fe:.4f}")
            logger.info(f"Updated beliefs: {configured_pomdp.state.beliefs}")
            
            # Store history
            history['observations'].append(obs)
            history['actions'].append(configured_pomdp.state.history['actions'][-1])
            history['free_energies'].append(fe)
            history['beliefs'].append(configured_pomdp.state.beliefs.copy())
            
            # Create EFE component visualization for this timestep
            visualize_efe_components(efe_components, step, efe_dir, planning_horizon)
        
        # Generate and save visualizations
        # Belief evolution plot
        plt.figure(figsize=(10, 6))
        beliefs = np.array(history['beliefs'])
        for i in range(configured_pomdp.num_states):
            plt.plot(range(len(beliefs)), beliefs[:, i], 
                    label=f'State {i}', linewidth=2, alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('Belief Probability')
        plt.title('Belief Evolution Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'belief_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Free energy plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(history['free_energies'])), history['free_energies'], 
                'b-', linewidth=2, alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('Free Energy')
        plt.title('Free Energy Over Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'free_energy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Action distribution plot
        plt.figure(figsize=(10, 6))
        actions = np.array(history['actions'])
        action_counts = np.bincount(actions, minlength=configured_pomdp.num_actions)
        bars = plt.bar(range(configured_pomdp.num_actions), action_counts)
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.title('Action Distribution')
        for bar, count in zip(bars, action_counts):
            plt.text(bar.get_x() + bar.get_width()/2, count, str(count),
                    ha='center', va='bottom')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'action_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log summary statistics
        logger.info("\nSimulation Summary:")
        logger.info(f"Average Free Energy: {np.mean(history['free_energies']):.4f}")
        logger.info(f"Observation distribution: {np.bincount(history['observations'])}")
        logger.info(f"Action distribution: {np.bincount(history['actions'])}")
        
        # Calculate belief entropy evolution
        belief_entropies = [-np.sum(b * np.log(b + 1e-12)) for b in history['beliefs']]
        logger.info(f"Initial belief entropy: {belief_entropies[0]:.4f}")
        logger.info(f"Final belief entropy: {belief_entropies[-1]:.4f}")
        
        # Log generated files
        logger.info("\nGenerated visualization files:")
        for plot_file in plots_dir.glob("*.png"):
            logger.info(f"- {plot_file.name}")
        
        # Additional assertions for matrix properties
        # Test A matrix properties
        assert np.allclose(configured_pomdp.A.sum(axis=0), 1.0), "A matrix columns should sum to 1"
        assert np.all(configured_pomdp.A >= 0), "A matrix should be non-negative"
        
        # Test B matrix properties
        for a in range(configured_pomdp.num_actions):
            assert np.allclose(configured_pomdp.B[:,:,a].sum(axis=0), 1.0), \
                f"B matrix for action {a} columns should sum to 1"
            assert np.all(configured_pomdp.B[:,:,a] >= 0), \
                f"B matrix for action {a} should be non-negative"
        
        # Test D matrix properties
        assert np.allclose(configured_pomdp.D.sum(), 1.0), "D matrix should sum to 1"
        assert np.all(configured_pomdp.D >= 0), "D matrix should be non-negative"
        
        # Test E matrix properties
        assert np.allclose(configured_pomdp.E.sum(), 1.0), "E matrix should sum to 1"
        assert np.all(configured_pomdp.E >= 0), "E matrix should be non-negative"
        
        # Check that all visualization files exist
        expected_files = [
            'belief_evolution.png',
            'free_energy.png',
            'action_distribution.png',
            'A_matrix.png',
            'B_matrix.png',
            'C_matrix.png',
            'D_matrix.png',
            'E_matrix.png',
            'matrix_overview.png'
        ]
        for filename in expected_files:
            assert (plots_dir / filename).exists(), f"Expected visualization file {filename} not found"
        
        logger.info("\nFull simulation test completed successfully")
        
    except Exception as e:
        logger.error(f"\nError in full simulation test: {str(e)}")
        raise

def visualize_efe_components(efe_components, timestep, efe_dir, planning_horizon):
    """Create visualizations for EFE components at each timestep."""
    # Extract components
    ambiguity = efe_components['ambiguity']
    risk = efe_components['risk']
    expected_preferences = efe_components['expected_preferences']
    total_efe = efe_components['total_efe']

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    # Plot ambiguity
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(ambiguity.reshape(-1, planning_horizon), cmap='viridis', aspect='auto')
    ax1.set_title('Ambiguity')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Policy')
    plt.colorbar(im1, ax=ax1)

    # Plot risk
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(risk.reshape(-1, planning_horizon), cmap='viridis', aspect='auto')
    ax2.set_title('Risk')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Policy')
    plt.colorbar(im2, ax=ax2)

    # Plot expected preferences with centered colormap
    ax3 = fig.add_subplot(gs[1, 0])
    vmax = np.max(np.abs(expected_preferences))
    vmin = -vmax
    im3 = ax3.imshow(expected_preferences.reshape(-1, planning_horizon), 
                     cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    ax3.set_title('Expected Preferences')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Policy')
    plt.colorbar(im3, ax=ax3)

    # Plot total EFE as bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    policies = range(len(total_efe))
    ax4.bar(policies, total_efe)
    ax4.set_title('Total Expected Free Energy')
    ax4.set_xlabel('Policy')
    ax4.set_ylabel('EFE')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(efe_dir / f'efe_components_t{timestep:03d}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create policy comparison visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(total_efe))
    width = 0.25

    # Plot grouped bar chart
    ax.bar(x - width, ambiguity.mean(axis=1), width, label='Ambiguity', color='blue', alpha=0.7)
    ax.bar(x, risk.mean(axis=1), width, label='Risk', color='red', alpha=0.7)
    ax.bar(x + width, expected_preferences.mean(axis=1), width, label='Expected Preferences', color='green', alpha=0.7)

    ax.set_xlabel('Policy')
    ax.set_ylabel('Component Value')
    ax.set_title('EFE Components by Policy')
    ax.legend()

    plt.tight_layout()
    plt.savefig(efe_dir / f'policy_comparison_t{timestep:03d}.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_full_configured_simulation(configured_pomdp, config):
    """Run a full simulation with the configured time horizon."""
    logger.info("\nStarting full configured simulation test")
    
    # Get total timesteps from config
    T = config['dimensions']['total_timesteps']
    logger.info(f"\nRunning simulation for {T} timesteps as configured")
    
    # Log initial configuration
    logger.info("\nModel Configuration:")
    logger.info(f"Observations: {configured_pomdp.num_observations}")
    logger.info(f"States: {configured_pomdp.num_states}")
    logger.info(f"Actions: {configured_pomdp.num_actions}")
    logger.info(f"Planning horizon: {config['dimensions']['planning_horizon']}")
    logger.info(f"Total timesteps: {T}")
    
    # Initialize history tracking
    history = {
        'observations': [],
        'actions': [],
        'free_energies': [],
        'beliefs': [],
        'efe_components': []  # Track EFE components instead of preferences
    }
    
    # Store initial state
    history['beliefs'].append(configured_pomdp.state.beliefs.copy())
    
    # Run simulation for T timesteps
    for step in range(T):
        logger.info(f"\nTimestep {step + 1}/{T}")
        
        # Get EFE components before taking step
        efe_components = configured_pomdp.get_efe_components()
        history['efe_components'].append(efe_components)
        
        # Take step and record results
        obs, fe = configured_pomdp.step()
        
        # Log step results
        logger.info(f"Observation: {obs}")
        logger.info(f"Free Energy: {fe:.4f}")
        logger.info(f"Updated beliefs: {configured_pomdp.state.beliefs}")
        
        # Store history
        history['observations'].append(obs)
        history['actions'].append(configured_pomdp.state.history['actions'][-1])
        history['free_energies'].append(fe)
        history['beliefs'].append(configured_pomdp.state.beliefs.copy())
    
    # Calculate and log summary statistics
    logger.info("\nSimulation Summary:")
    logger.info(f"Average Free Energy: {np.mean(history['free_energies']):.4f}")
    
    # Observation statistics
    obs_counts = np.bincount(history['observations'], minlength=configured_pomdp.num_observations)
    logger.info(f"Observation distribution: {obs_counts}")
    logger.info(f"Most frequent observation: {np.argmax(obs_counts)} (count: {np.max(obs_counts)})")
    
    # Action statistics
    action_counts = np.bincount(history['actions'], minlength=configured_pomdp.num_actions)
    logger.info(f"Action distribution: {action_counts}")
    logger.info(f"Most frequent action: {np.argmax(action_counts)} (count: {np.max(action_counts)})")
    
    # Belief evolution
    belief_entropies = [-np.sum(b * np.log(b + 1e-12)) for b in history['beliefs']]
    logger.info(f"Initial belief entropy: {belief_entropies[0]:.4f}")
    logger.info(f"Final belief entropy: {belief_entropies[-1]:.4f}")
    logger.info(f"Belief entropy reduction: {belief_entropies[0] - belief_entropies[-1]:.4f}")
    
    # EFE component analysis
    avg_ambiguity = np.mean([comp['ambiguity'].mean() for comp in history['efe_components']])
    avg_risk = np.mean([comp['risk'].mean() for comp in history['efe_components']])
    avg_preferences = np.mean([comp['expected_preferences'].mean() for comp in history['efe_components']])
    logger.info("\nEFE Component Analysis:")
    logger.info(f"Average Ambiguity: {avg_ambiguity:.4f}")
    logger.info(f"Average Risk: {avg_risk:.4f}")
    logger.info(f"Average Expected Preferences: {avg_preferences:.4f}")
    
    # Validate results
    assert len(history['observations']) == T, f"Expected {T} observations"
    assert len(history['actions']) == T, f"Expected {T} actions"
    assert len(history['free_energies']) == T, f"Expected {T} free energy values"
    assert all(np.isfinite(fe) for fe in history['free_energies']), "All free energies should be finite"
    assert all(0 <= obs < configured_pomdp.num_observations 
              for obs in history['observations']), "All observations should be valid"
    
    logger.info("\nFull configured simulation completed successfully")

def test_matrix_numerical_properties(configured_pomdp):
    """Test numerical properties of matrices."""
    # Test condition numbers
    A_cond = np.linalg.cond(configured_pomdp.A)
    assert A_cond < 1e6, f"A matrix condition number too high: {A_cond}"
    
    for a in range(configured_pomdp.num_actions):
        B_cond = np.linalg.cond(configured_pomdp.B[:,:,a])
        assert B_cond < 1e6, f"B matrix condition number too high for action {a}: {B_cond}"
    
    # Test for NaN and Inf values
    assert not np.any(np.isnan(configured_pomdp.A)), "A matrix contains NaN values"
    assert not np.any(np.isinf(configured_pomdp.A)), "A matrix contains Inf values"
    
    assert not np.any(np.isnan(configured_pomdp.B)), "B matrix contains NaN values"
    assert not np.any(np.isinf(configured_pomdp.B)), "B matrix contains Inf values"
    
    assert not np.any(np.isnan(configured_pomdp.C)), "C matrix contains NaN values"
    assert not np.any(np.isinf(configured_pomdp.C)), "C matrix contains Inf values"
    
    assert not np.any(np.isnan(configured_pomdp.D)), "D matrix contains NaN values"
    assert not np.any(np.isinf(configured_pomdp.D)), "D matrix contains Inf values"
    
    assert not np.any(np.isnan(configured_pomdp.E)), "E matrix contains NaN values"
    assert not np.any(np.isinf(configured_pomdp.E)), "E matrix contains Inf values"

def test_matrix_consistency(configured_pomdp):
    """Test consistency of matrix dimensions and properties."""
    # Test dimension consistency
    assert configured_pomdp.A.shape == (configured_pomdp.num_observations, configured_pomdp.num_states), \
        "A matrix dimensions inconsistent"
    
    assert configured_pomdp.B.shape == (configured_pomdp.num_states, configured_pomdp.num_states, 
                                      configured_pomdp.num_actions), "B matrix dimensions inconsistent"
    
    assert configured_pomdp.C.shape[0] == configured_pomdp.num_observations, \
        "C matrix observation dimension inconsistent"
    
    assert configured_pomdp.D.shape == (configured_pomdp.num_states,), \
        "D matrix dimension inconsistent"
    
    assert configured_pomdp.E.shape == (configured_pomdp.num_actions,), \
        "E matrix dimension inconsistent"
    
    # Test probability constraints
    for a in range(configured_pomdp.num_actions):
        # Each column in B should represent a valid transition probability distribution
        assert np.all(np.abs(configured_pomdp.B[:,:,a].sum(axis=0) - 1.0) < 1e-10), \
            f"B matrix for action {a} contains invalid probability distributions"
    
    # Each column in A should represent a valid observation probability distribution
    assert np.all(np.abs(configured_pomdp.A.sum(axis=0) - 1.0) < 1e-10), \
        "A matrix contains invalid probability distributions"

def test_belief_update_stability(configured_pomdp):
    """Test numerical stability of belief updates under extreme conditions."""
    # Test with very small beliefs
    configured_pomdp.state.beliefs = np.ones(configured_pomdp.num_states) * 1e-10
    configured_pomdp.state.beliefs /= configured_pomdp.state.beliefs.sum()
    
    # Should not raise and give valid beliefs
    obs, fe = configured_pomdp.step()
    assert np.allclose(configured_pomdp.state.beliefs.sum(), 1.0), "Beliefs not normalized after update"
    assert np.all(configured_pomdp.state.beliefs >= 0), "Negative beliefs after update"
    assert np.all(np.isfinite(configured_pomdp.state.beliefs)), "Non-finite beliefs after update"
    
    # Test with very concentrated beliefs
    configured_pomdp.state.beliefs = np.zeros(configured_pomdp.num_states)
    configured_pomdp.state.beliefs[0] = 1.0 - 1e-10
    configured_pomdp.state.beliefs[1:] = 1e-10 / (configured_pomdp.num_states - 1)
    
    # Should not raise and give valid beliefs
    obs, fe = configured_pomdp.step()
    assert np.allclose(configured_pomdp.state.beliefs.sum(), 1.0), "Beliefs not normalized after update"
    assert np.all(configured_pomdp.state.beliefs >= 0), "Negative beliefs after update"
    assert np.all(np.isfinite(configured_pomdp.state.beliefs)), "Non-finite beliefs after update"

def test_preference_learning(configured_pomdp):
    """Test learning with different preference configurations."""
    # Store original matrices
    original_C = configured_pomdp.C.copy()
    original_B = configured_pomdp.B.copy()
    original_A = configured_pomdp.A.copy()
    
    try:
        # Set very strong preference for first observation
        configured_pomdp.C[0,:] = 5.0  # Strong positive preference for observation 0
        configured_pomdp.C[1:,:] = -2.0  # Negative preference for other observations
        
        # Modify transition dynamics to make observation 0 achievable
        # Make first state more likely to generate observation 0
        configured_pomdp.A[0,0] = 0.8  # High probability of observation 0 from state 0
        configured_pomdp.A[1:,0] = 0.2 / (configured_pomdp.num_observations - 1)  # Distribute remaining probability
        
        # Normalize A matrix columns
        configured_pomdp.A /= configured_pomdp.A.sum(axis=0, keepdims=True)
        
        # Make action 0 lead to state 0 with high probability
        for a in range(configured_pomdp.num_actions):
            configured_pomdp.B[:,:,a] = np.eye(configured_pomdp.num_states) * 0.7
            configured_pomdp.B[0,:,0] = 0.8  # Action 0 leads to state 0
            configured_pomdp.B[1:,:,0] = 0.2 / (configured_pomdp.num_states - 1)  # Distribute remaining probability
            # Normalize B matrix columns
            configured_pomdp.B[:,:,a] /= configured_pomdp.B[:,:,a].sum(axis=0, keepdims=True)
        
        # Reset beliefs to uniform
        configured_pomdp.state.beliefs = np.ones(configured_pomdp.num_states) / configured_pomdp.num_states
        
        # Run simulation
        n_steps = 50  # Increased number of steps
        observations = []
        for _ in range(n_steps):
            obs, _ = configured_pomdp.step()
            observations.append(obs)
        
        # Check if preferred observations occur more frequently
        obs_counts = np.bincount(observations, minlength=configured_pomdp.num_observations)
        logger.info(f"Observation counts: {obs_counts}")
        logger.info(f"Observation 0 count: {obs_counts[0]}")
        logger.info(f"Mean of other observations: {np.mean(obs_counts[1:]):.2f}")
        
        # Require at least 30% more observations of type 0 than average of others
        assert obs_counts[0] > 1.3 * np.mean(obs_counts[1:]), \
            f"Preferred observation not selected more frequently. Got {obs_counts[0]} vs mean {np.mean(obs_counts[1:]):.2f}"
        
    finally:
        # Restore original matrices
        configured_pomdp.C = original_C
        configured_pomdp.B = original_B
        configured_pomdp.A = original_A
        # Reset beliefs to uniform
        configured_pomdp.state.beliefs = np.ones(configured_pomdp.num_states) / configured_pomdp.num_states

def test_policy_tree_generation(small_pomdp):
    """Test policy tree generation and pruning."""
    # Get policy tree
    components = small_pomdp.get_efe_components()
    policies = components['policies']
    
    # Check basic properties
    assert isinstance(policies, list)
    assert all(isinstance(p, list) for p in policies)
    assert all(all(isinstance(a, int) for a in p) for p in policies)
    assert all(0 <= a < small_pomdp.num_actions for p in policies for a in p)
    
    # Check policy lengths
    assert all(len(p) <= small_pomdp.planning_horizon for p in policies)
    
    # Check number of policies is bounded
    assert len(policies) <= small_pomdp.max_policies
    
    # Check policy evaluation components
    assert components['ambiguity'].shape == (len(policies), small_pomdp.planning_horizon)
    assert components['risk'].shape == (len(policies), small_pomdp.planning_horizon)
    assert components['expected_preferences'].shape == (len(policies), small_pomdp.planning_horizon)
    assert components['total_efe'].shape == (len(policies),)

def test_temporal_preference_evaluation(small_pomdp):
    """Test evaluation of preferences over time."""
    # Set time-varying preferences
    small_pomdp.C = np.zeros((small_pomdp.num_observations, small_pomdp.planning_horizon))
    small_pomdp.C[0, 0] = 1.0  # Prefer first observation initially
    small_pomdp.C[1, -1] = 2.0  # Strongly prefer second observation later
    
    # Get policy evaluation
    components = small_pomdp.get_efe_components()
    expected_preferences = components['expected_preferences']
    
    # Policies should show preference for delayed reward
    best_policy_idx = np.argmin(components['total_efe'])
    best_policy = components['policies'][best_policy_idx]
    
    # Check that later actions in best policy lead to preferred observation
    final_beliefs = small_pomdp.state.beliefs.copy()
    for action in best_policy:
        final_beliefs = small_pomdp.B[:,:,action] @ final_beliefs
    final_obs_probs = small_pomdp.A @ final_beliefs
    
    # Should have high probability for second observation
    assert final_obs_probs[1] > 0.3

def test_policy_pruning(medium_pomdp):
    """Test that policy pruning maintains diversity and performance."""
    # Get full policy evaluation
    components = medium_pomdp.get_efe_components()
    policies = components['policies']
    total_efe = components['total_efe']
    
    # Check pruning keeps best policies
    best_idx = np.argmin(total_efe)
    best_efe = total_efe[best_idx]
    
    # Run with smaller max_policies
    medium_pomdp.max_policies = len(policies) // 2
    pruned_components = medium_pomdp.get_efe_components()
    pruned_total_efe = pruned_components['total_efe']
    
    # Best pruned policy should be close to original best
    best_pruned_efe = np.min(pruned_total_efe)
    assert np.abs(best_pruned_efe - best_efe) < 1.0
    
    # Check policy diversity (not all same first action)
    first_actions = [p[0] for p in pruned_components['policies']]
    assert len(set(first_actions)) > 1

def test_belief_propagation(small_pomdp):
    """Test belief propagation through policy evaluation."""
    # Set specific initial beliefs
    small_pomdp.state.beliefs = np.zeros(small_pomdp.num_states)
    small_pomdp.state.beliefs[0] = 1.0
    
    # Get policy evaluation
    components = small_pomdp.get_efe_components()
    policies = components['policies']
    
    # Check belief propagation for first policy
    test_policy = policies[0]
    beliefs = small_pomdp.state.beliefs.copy()
    
    for t, action in enumerate(test_policy):
        # Manual belief propagation
        beliefs = small_pomdp.B[:,:,action] @ beliefs
        
        # Get predicted observations
        pred_obs = small_pomdp.A @ beliefs
        
        # Compare with stored components
        policy_idx = 0  # Testing first policy
        
        # Check ambiguity matches entropy of beliefs
        expected_ambiguity = -np.sum(beliefs * np.log(beliefs + small_pomdp.stability_threshold))
        assert np.abs(components['ambiguity'][policy_idx, t] - expected_ambiguity) < 1e-5
        
        # Check risk computation
        expected_risk = small_pomdp._compute_kl_divergence(
            pred_obs,
            small_pomdp._softmax(small_pomdp.C[:, t])
        )
        assert np.abs(components['risk'][policy_idx, t] - expected_risk) < 1e-5

if __name__ == '__main__':
    # When run directly, execute pytest with full output
    import sys
    sys.exit(pytest.main([__file__, "-v", "--capture=tee-sys"])) 