---
title: API Implementation Guide
type: guide
status: stable
created: 2025-01-01
updated: 2026-02-07
tags:
  - api
  - implementation
  - active_inference
  - integration
semantic_relations:
  - type: relates_to
    links:
      - [[docs/api/api_reference]]
      - [[docs/implementation/README]]
      - [[docs/guides/best_practices]]
---

# API Implementation Guide

## Overview

This guide explains how to implement and consume the Cognitive Framework APIs. The APIs provide programmatic access to Active Inference agent creation, environment management, and simulation execution.

## Core API Patterns

### Agent Creation API

```python
from cognitive.agents import create_agent

# Create an agent with a specified generative model
agent = create_agent(
    model_type="discrete_pomdp",
    num_states=4,
    num_observations=4,
    num_actions=3,
    planning_horizon=2,
    precision=16.0  # Inverse temperature for action selection
)

# Update beliefs from observation
posterior = agent.infer_states(observation=obs)

# Select action
action = agent.select_action()
```

### Environment API

```python
from cognitive.environments import create_environment

# Create a grid environment
env = create_environment(
    env_type="grid_world",
    size=(10, 10),
    rewards={"food": (5, 5), "nest": (0, 0)},
    dynamics="deterministic"
)

# Step through the environment
observation, reward, done, info = env.step(action)
```

### Simulation Loop API

```python
from cognitive.simulation import SimulationRunner

runner = SimulationRunner(
    agent=agent,
    environment=env,
    num_episodes=100,
    max_steps_per_episode=50,
    log_level="INFO"
)

# Run and collect results
results = runner.run()
print(f"Mean reward: {results.mean_reward:.3f}")
print(f"Mean free energy: {results.mean_free_energy:.3f}")
```

## Integration Patterns

### Custom Generative Models

To implement a custom generative model, subclass `GenerativeModel`:

```python
from cognitive.models import GenerativeModel

class CustomModel(GenerativeModel):
    """Custom generative model with domain-specific structure."""

    def likelihood(self, observation, state):
        """P(o|s): probability of observation given state."""
        return self.A[observation, state]

    def transition(self, state, action):
        """P(s'|s, a): state transition probability."""
        return self.B[:, state, action]

    def preference(self, observation):
        """log P(o): log prior preference over observations."""
        return np.log(self.C[observation] + 1e-16)
```

### Error Handling

Always handle errors gracefully following the framework conventions:

```python
from cognitive.exceptions import (
    ModelSpecificationError,
    InferenceError,
    ConvergenceWarning
)

try:
    posterior = agent.infer_states(obs)
except InferenceError as e:
    logger.error(f"Inference failed: {e}")
    posterior = agent.prior  # Fall back to prior
except ConvergenceWarning as w:
    logger.warning(f"Inference did not converge: {w}")
```

### Typed Configurations

Use typed configurations for all model parameters:

```python
from dataclasses import dataclass

@dataclass
class AgentConfig:
    """Typed configuration for Active Inference agents."""
    num_states: int
    num_observations: int
    num_actions: int
    planning_horizon: int = 1
    precision: float = 16.0
    learning_rate: float = 0.1
    
    def validate(self):
        assert self.num_states > 0, "Must have at least one state"
        assert self.precision > 0, "Precision must be positive"
```

## Best Practices

1. **Use typed configurations** — define explicit types for all parameters
2. **Handle errors gracefully** — catch specific exceptions and provide fallbacks
3. **Specify precision parameters** — always set precision explicitly rather than relying on defaults
4. **Log inference steps** — record beliefs, free energy, and actions for debugging
5. **Validate inputs** — check matrix dimensions and normalization before inference

## API Reference

For the complete API reference, see [[docs/api/api_reference|API Reference]].

## Related Resources

- [[docs/implementation/README|Implementation Guides]] — deeper implementation patterns
- [[docs/guides/best_practices|Best Practices]] — coding standards and conventions
- [[docs/guides/agent_development|Agent Development Guide]] — agent architecture guide
- [[docs/api/api_reference|API Reference]] — complete API documentation
