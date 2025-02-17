# Generic POMDP Implementation

This is a comprehensive implementation of a Partially Observable Markov Decision Process (POMDP) using Active Inference principles. The implementation is designed to be flexible, numerically stable, and suitable for a wide range of applications.

## Key Features

- **Belief Updating**: Robust variational inference with momentum and adaptive learning rates
- **Action Selection**: Expected Free Energy minimization with temporal horizon planning
- **Preference Learning**: Temporal preference encoding and updating
- **Numerical Stability**: Comprehensive handling of edge cases and numerical issues
- **State Management**: Full state saving/loading capabilities
- **Visualization**: Rich visualization tools for beliefs, policies, and EFE components

## Core Components

### Matrix Representations

- [[A_matrix]] (Observation Model):
  - Dimensions: `(num_observations, num_states)`
  - Properties: Column stochastic (sums to 1 along observations)
  - Represents: P(o|s) - probability of observations given states

- **B Matrix** (Transition Model):
  - Dimensions: `(num_states, num_states, num_actions)`
  - Properties: Column stochastic per action
  - Represents: P(s'|s,a) - state transitions under actions

- **C Matrix** (Preferences):
  - Dimensions: `(num_observations, planning_horizon)`
  - Properties: Real-valued preferences
  - Represents: Log probabilities of preferred observations

- **D Matrix** (Initial Beliefs):
  - Dimensions: `(num_states,)`
  - Properties: Normalized probability distribution
  - Represents: Prior beliefs over states

- **E Matrix** (Policy Prior):
  - Dimensions: `(num_actions,)`
  - Properties: Normalized probability distribution
  - Represents: Prior preferences over actions

### Expected Free Energy Components

The implementation computes several components for policy evaluation:

1. **Ambiguity** (Epistemic Value):
   - Measures uncertainty in beliefs
   - Drives information-seeking behavior

2. **Risk**:
   - KL divergence between predicted and preferred observations
   - Ensures policy alignment with preferences

3. **Expected Preferences**:
   - Direct value of anticipated observations
   - Guides goal-directed behavior

## Usage

```python
from generic_pomdp import GenericPOMDP

# Initialize POMDP
pomdp = GenericPOMDP(
    num_observations=4,
    num_states=3,
    num_actions=2,
    planning_horizon=4
)

# Take a step with automatic action selection
observation, free_energy = pomdp.step()

# Take a step with specific action
observation, free_energy = pomdp.step(action=1)

# Get EFE components for analysis
components = pomdp.get_efe_components()

# Save/load state
pomdp.save_state("pomdp_state.json")
pomdp.load_state("pomdp_state.json")
```

## Implementation Details

### Belief Updating

The belief updating mechanism uses several techniques for robustness:
- Momentum-based optimization
- Adaptive learning rates
- Numerical stability thresholds
- Convergence monitoring

### Action Selection

Action selection is performed by:
1. Generating possible policies up to planning horizon
2. Computing EFE components for each policy
3. Applying softmax with temperature parameter
4. Sampling from resulting distribution

### Numerical Stability

The implementation includes comprehensive numerical stability features:
- Minimum thresholds for probabilities
- Safe logarithm computations
- Normalized intermediate computations
- Bounded preference values

## Testing

Comprehensive test suite available in `test_generic_pomdp.py` covering:
- Matrix properties and initialization
- Belief updating mechanics
- Action selection
- Numerical stability
- State saving/loading
- Full simulation scenarios

## Visualization

The implementation includes visualization tools for:
- Belief evolution over time
- Policy evaluation components
- Free energy landscapes
- Preference learning dynamics

## Requirements

- NumPy
- JSON (for state saving/loading)
- Logging

## References

For theoretical background:
- Active Inference: A Process Theory (Friston et al.)
- The Free Energy Principle (Friston)
- Information Theory of Decisions and Actions (Tishby & Polani)

## License

MIT License - See LICENSE file for details 