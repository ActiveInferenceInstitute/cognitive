# Continuous Generic Active Inference

This implementation provides a continuous-time, continuous-state space active inference agent. The agent learns and operates in continuous domains using modern deep learning techniques while adhering to active inference principles.

## Overview

The implementation consists of several key components:

1. **Generative Model**: A neural network that learns:

   - Observation model p(o|s)

   - State transition dynamics p(s'|s,a)

1. **Recognition Model**: A neural network that implements:

   - Approximate posterior q(s|o)

   - Belief updating through variational inference

1. **Action Selection**: Continuous action selection through:

   - Expected Free Energy minimization

   - Gradient-based optimization

1. **Visualization**: Comprehensive visualization tools for:

   - Belief evolution

   - Free energy landscapes

   - Phase space trajectories

   - Action distributions

## Mathematical Framework

The implementation follows the active inference framework in continuous time:

1. **Free Energy**:

   ```text

   F = ⟨ln q(s,θ) - ln p(o,s,θ)⟩_q

   ```

1. **Belief Updating**:

   ```text

   ∂_t q(s) = -∂_s F[q]

   ```

1. **Action Selection**:

   ```text

   a = -∂_a F

   ```

## Installation

1. Clone the repository

1. Install dependencies:

   ```bash

   pip install -r requirements.txt

   ```

## Usage

Basic usage example:

```python

from continuous_generic import ContinuousActiveInference

from visualization import ContinuousVisualizer

# Initialize agent

agent = ContinuousActiveInference(

    state_dim=2,

    obs_dim=2,

    action_dim=1

)

# Initialize visualizer

visualizer = ContinuousVisualizer("Output")

# Run simulation

obs = initial_observation

for t in range(100):

    # Take step

    action, free_energy = agent.step(obs)

    # Get next observation from environment

    obs = environment_step(action)

    # Visualize (every 10 steps)

    if t % 10 == 0:

        visualizer.create_summary_plot(

            agent.state.history,

            f"Output/summary_t{t:03d}.png"

        )

```

## Configuration

The behavior can be configured through `configuration.yaml`:

```yaml

model:

  state_dim: 2          # State space dimension

  obs_dim: 2           # Observation space dimension

  action_dim: 1        # Action space dimension

  hidden_dim: 64       # Neural network hidden dimension

  dt: 0.01            # Integration time step

  learning_rate: 1e-3  # Optimization learning rate

  temperature: 1.0     # Action selection temperature

```

## Implementation Details

### Generative Model

The generative model uses deep neural networks to learn:

1. Observation mapping from states to observations

1. State transition dynamics under actions

```python

class GenerativeModel(nn.Module):

    def __init__(self, state_dim, obs_dim):

        self.obs_net = nn.Sequential(...)    # p(o|s)

        self.dynamics_net = nn.Sequential(...) # p(s'|s,a)

```

### Recognition Model

The recognition model implements variational inference:

1. Approximate posterior over states

1. Belief updating through gradient descent

```python

class RecognitionModel(nn.Module):

    def __init__(self, obs_dim, state_dim):

        self.encoder = nn.Sequential(...)  # q(s|o)

```

### Action Selection

Actions are selected by minimizing expected free energy:

1. Predict future observations

1. Compute expected free energy gradients

1. Optimize actions through gradient descent

## Visualization

The visualization module provides tools for analyzing agent behavior:

1. **Belief Evolution**: Track belief updates over time

1. **Free Energy**: Monitor optimization progress

1. **Phase Space**: Visualize belief trajectories

1. **Action Evolution**: Analyze action selection

## Testing

Comprehensive test suite available:

```bash

pytest test_continuous_generic.py

```

Tests cover:

1. Model initialization

1. Belief updating

1. Action selection

1. Visualization

1. State saving/loading

## Contributing

1. Fork the repository

1. Create feature branch

1. Commit changes

1. Push to branch

1. Create pull request

## License

MIT License

## References

1. Friston, K. J., et al. (2017). Active inference, curiosity and insight.

1. Buckley, C. L., et al. (2017). The free energy principle for action and perception: A mathematical review.

1. Tschantz, A., et al. (2020). Learning action-oriented models through active inference.

