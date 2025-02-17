# SimplePOMDP

A simple implementation of a Partially Observable Markov Decision Process (POMDP) using Active Inference principles. This implementation demonstrates how Active Inference can be used for decision-making and belief updating in partially observable environments.

## Overview

The SimplePOMDP implementation includes:
- A configurable state space with partial observability
- Action selection using Active Inference principles
- Belief updating based on observations and actions
- Visualization tools for analyzing model behavior
- Comprehensive test suite

## Structure

```
SimplePOMDP/
├── README.md              # This file
├── configuration.yaml     # Model configuration
├── simple_pomdp.py       # Main implementation
└── tests/
    └── test_simple_pomdp.py  # Test suite
```

## Features

### State Space
- Configurable number of states
- Partial observability through observation matrix
- Belief state tracking and updating

### Actions
- Configurable action space
- Action selection using Expected Free Energy
- Policy evaluation and selection

### Matrices
1. **A Matrix** (Observation/Likelihood)
   - Maps states to observations
   - Column stochastic (probabilities sum to 1)
   - Configurable initialization

2. **B Matrix** (Transition/Dynamics)
   - State transition probabilities for each action
   - Row stochastic for each action
   - Supports identity-based initialization

3. **C Matrix** (Preferences)
   - Defines preferred observations
   - Supports goal-directed behavior
   - Time-horizon dependent preferences

4. **D Matrix** (Prior Beliefs)
   - Initial belief distribution over states
   - Uniform or custom initialization
   - Ensures valid probability distribution

5. **E Matrix** (Policies)
   - Defines available action sequences
   - Supports various policy types
   - Used for action selection

### Visualization
- Belief evolution over time
- Free energy landscape visualization
- Policy evaluation plots
- State transition diagrams
- Observation likelihood heatmaps

## Configuration

The model is configured through a YAML file with the following sections:

```yaml
model:
  name: "SimplePOMDP"
  description: "..."
  version: "0.1.0"

state_space:
  num_states: 3
  state_labels: [...]
  initial_state: 0

observation_space:
  num_observations: 2
  observation_labels: [...]

action_space:
  num_actions: 2
  action_labels: [...]

matrices:
  A_matrix: {...}
  B_matrix: {...}
  C_matrix: {...}
  D_matrix: {...}
  E_matrix: {...}

inference:
  time_horizon: 5
  num_iterations: 10
  learning_rate: 0.1
  temperature: 1.0

visualization:
  output_dir: "..."
  formats: ["png"]
  style: {...}
```

## Usage

### Basic Usage

```python
from Things.SimplePOMDP.simple_pomdp import SimplePOMDP

# Initialize the model
model = SimplePOMDP("path/to/config.yaml")

# Take a step with automatic action selection
observation, free_energy = model.step()

# Take a step with specified action
observation, free_energy = model.step(action=0)

# Generate visualizations
model.visualize("belief_evolution")
model.visualize("free_energy_landscape")
model.visualize("policy_evaluation")
```

### Running Tests

```bash
# Run all tests
pytest tests/test_simple_pomdp.py

# Run specific test
pytest tests/test_simple_pomdp.py::test_initialization

# Run with coverage
pytest tests/test_simple_pomdp.py --cov=Things.SimplePOMDP
```

## Visualization Outputs

All visualization outputs are saved to the configured output directory. Available plots include:

1. **Belief Evolution**
   - Shows how beliefs about states change over time
   - Line plot with one line per state
   - Saved as `belief_evolution.png`

2. **Free Energy Landscape**
   - 3D surface plot of expected free energy
   - Shows relationship between beliefs and free energy
   - Saved as `free_energy_landscape.png`

3. **Policy Evaluation**
   - Bar plot of expected free energy for each policy
   - Helps understand action selection
   - Saved as `policy_evaluation.png`

4. **State Transitions**
   - Heatmap of transition probabilities for each action
   - Shows dynamics of the environment
   - Saved as `state_transitions.png`

5. **Observation Likelihood**
   - Heatmap of observation probabilities given states
   - Shows observation model structure
   - Saved as `observation_likelihood.png`

6. **Belief History**
   - Heatmap visualization of belief evolution
   - Shows complete belief history over time
   - Color intensity indicates belief strength
   - Saved as `belief_history.png`

7. **Free Energy History**
   - Line plot of free energy values over time
   - Includes trend line for convergence analysis
   - Saved as `free_energy_history.png`

8. **Action History**
   - Scatter plot of selected actions over time
   - Color gradient shows temporal progression
   - Includes jittering for better visibility
   - Saved as `action_history.png`

9. **State Distribution**
   - Combined visualization of:
     - Current belief distribution (bar plot)
     - Belief entropy history (line plot)
   - Saved as `state_distribution.png`

### Example Usage

```python
# Generate standard visualizations
model.visualize("belief_evolution")
model.visualize("free_energy_landscape")
model.visualize("policy_evaluation")

# Generate advanced visualizations
model.visualize("belief_history")
model.visualize("free_energy_history")
model.visualize("action_history")
model.visualize("state_distribution")
```

## Testing

The test suite provides comprehensive coverage of the implementation:

### Core Functionality Tests
1. **Initialization Tests**
   - Matrix shapes and types
   - Configuration validation
   - Initial state setup

2. **Functional Tests**
   - Step execution
   - Action selection
   - Belief updating
   - Free energy computation

3. **Visualization Tests**
   - Plot generation
   - File saving
   - Figure properties

4. **Validation Tests**
   - Matrix properties
   - Probability constraints
   - Error handling

### Advanced Tests
1. **Convergence Tests**
   - Belief convergence under consistent observations
   - Entropy reduction over time
   - Temporal consistency of state transitions

2. **Policy Tests**
   - Policy preference validation
   - Action selection distribution
   - Expected free energy computation

3. **Edge Case Tests**
   - Deterministic beliefs
   - Uniform beliefs
   - Extreme preferences
   - Learning rate sensitivity

4. **Visualization Property Tests**
   - Axis labels and titles
   - Data representation
   - Plot elements and styling

### Running Tests

```bash
# Run all tests
pytest tests/test_simple_pomdp.py

# Run specific test categories
pytest tests/test_simple_pomdp.py -k "test_belief"  # Belief-related tests
pytest tests/test_simple_pomdp.py -k "test_visualization"  # Visualization tests
pytest tests/test_simple_pomdp.py -k "test_edge"  # Edge case tests

# Run with coverage report
pytest tests/test_simple_pomdp.py --cov=Things.SimplePOMDP --cov-report=html
```

### Test Output
- Test results are displayed in the terminal
- Coverage report shows code coverage statistics
- Visualization outputs are saved to the configured directory
- Generated files are logged after test completion

## Implementation Details

### Active Inference Components

1. **Belief Updating**
   - Uses Bayes rule to update beliefs based on observations
   - Incorporates action effects through transition matrix
   - Maintains proper probability distributions

2. **Action Selection**
   - Computes expected free energy for each policy
   - Considers both ambiguity and risk
   - Uses softmax for action probabilities

3. **Free Energy Computation**
   - Includes both variational free energy and expected free energy
   - Accounts for preferences through C matrix
   - Handles multi-step policies

### Matrix Properties

All matrices maintain specific properties:

- **A Matrix**: Column stochastic, non-negative
- **B Matrix**: Row stochastic per action, non-negative
- **C Matrix**: Arbitrary values for preferences
- **D Matrix**: Sums to one, non-negative
- **E Matrix**: Integer values representing actions

## Testing

The test suite covers:

1. **Initialization Tests**
   - Matrix shapes and types
   - Configuration validation
   - Initial state setup

2. **Functional Tests**
   - Step execution
   - Action selection
   - Belief updating
   - Free energy computation

3. **Visualization Tests**
   - Plot generation
   - File saving
   - Figure properties

4. **Validation Tests**
   - Matrix properties
   - Probability constraints
   - Error handling

## Contributing

When contributing to this implementation:

1. Follow the project's code style
2. Add tests for new functionality
3. Update documentation
4. Ensure all tests pass
5. Verify visualization outputs

## Dependencies

- NumPy: Matrix operations and probability computations
- Matplotlib: Visualization generation
- PyYAML: Configuration file parsing
- pytest: Testing framework 