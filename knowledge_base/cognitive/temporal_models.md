---
title: Temporal Models
type: concept
status: stable
created: 2024-02-14
tags:
- cognitive
- temporal
- active_inference
- modeling
aliases:
  - temporal_dependencies
  - temporal_dynamics
semantic_relations:
  - type: implements
    links:
      - "[[active_inference]]"
  - type: relates
    links:
      - "[[generative_model]]"
      - "[[predictive_coding]]"
      - "[[hierarchical_inference]]"
      - "[[model_architecture]]"
      - "[[continuous_time_active_inference]]"
      - "[[model_selection]]"
      - "[[evidence_accumulation]]"
---


# Temporal Models

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

## Overview

Temporal models in [[active_inference|Active Inference]] represent how states and observations evolve over time. They are crucial for planning, prediction, and temporal inference across different timescales, from rapid sensorimotor processing to extended cognitive tasks.

## Core Concepts

### 1. Temporal Depth

- [[planning|Planning Horizon]]

- [[predictive_processing|Prediction Scope]]

- [[working_memory|Historical Context]]

- [[decision_making|Future Trajectories]]

### 2. Time Representations

- [[temporal_models|Discrete Timesteps]]

- [[continuous_time_active_inference|Continuous Dynamics]]

- Event Sequences

- [[state_transitions|State Transitions]]

### 3. Temporal Hierarchies

- [[sensorimotor_coordination|Fast Dynamics]] (lower levels)

- [[cognitive_control|Slow Dynamics]] (higher levels)

- [[temporal_binding|Temporal Integration]]

- [[hierarchical_inference|Scale Separation]]

### 4. Processing Types

- [[perceptual_inference|Perceptual Processing]]

- [[action_selection|Action Selection]]

- [[learning_mechanisms|Learning]]

- [[meta_learning|Meta-Learning]]

## Mathematical Framework

### 1. Discrete Time Models

```math

P(o₁,...,oₜ,s₁,...,sₜ) = P(o₁|s₁)P(s₁)∏ₖ₌₂ᵗ P(oₖ|sₖ)P(sₖ|sₖ₋₁)

```

where:

- oₖ are observations at time k

- sₖ are states at time k

- P(sₖ|sₖ₋₁) is the transition model

### 2. Continuous Time Models

```math

ds = f(s,a)dt + σdW

```

where:

- s is the state vector

- a is the action vector

- f(s,a) is the drift function

- σdW is stochastic diffusion

### 3. Hierarchical Time Models

```math

P(o,s₁,s₂,...,sₙ|τ₁,τ₂,...,τₙ) = P(o|s₁,τ₁)∏ᵢ P(sᵢ|sᵢ₊₁,τᵢ)

```

where:

- τᵢ represents timescale at level i

- sᵢ represents states at level i

- o represents observations

## Implementation

### 1. Basic Structure

```python
import numpy as np
rng = np.random.default_rng(0)

A = np.array([[0.9, 0.2], [0.1, 0.8]])        # P(o_k | s_k), illustrative
B = np.array([[0.95, 0.1], [0.05, 0.9]])      # P(s_k | s_{k-1}), columns sum to 1
D = np.array([0.5, 0.5])                      # P(s_0)

# Generate a sequence, then filter it: q(s_k) is proportional to P(o_k | s_k) sum_j P(s_k | j) q(j)
s, obs, beliefs, q = rng.choice(2, p=D), [], [], D
for k in range(20):
    s = rng.choice(2, p=B[:, s])
    o = rng.choice(2, p=A[:, s])
    q = A[o] * (B @ q)
    q = q / q.sum()
    obs.append(o); beliefs.append(q)
print(obs[:10], np.round(beliefs[-1], 2))
```

### 2. Advanced Components

#### Scale Separation

```python
import numpy as np

# Two levels with different timescales tau: the slow level integrates the fast one
dt, tau_fast, tau_slow = 0.01, 0.05, 1.0       # illustrative
fast, slow = 0.0, 0.0
for t in range(500):
    drive = np.sin(2 * np.pi * 5 * t * dt)     # 5 Hz input
    fast += dt / tau_fast * (drive - fast)     # tracks the input closely
    slow += dt / tau_slow * (fast - slow)      # smooths it over a longer window
print(round(fast, 2), round(slow, 2))
```

### 3. Model Comparison Framework

```python
import numpy as np

def log_evidence(obs, A, B, D):
    """ln P(o_1:K) by the forward algorithm with per-step normalisation."""
    q, total = D, 0.0
    for o in obs:
        joint = A[o] * (B @ q)
        total += np.log(joint.sum())
        q = joint / joint.sum()
    return total

A = np.array([[0.9, 0.2], [0.1, 0.8]])
D = np.array([0.5, 0.5])
sticky = np.array([[0.95, 0.1], [0.05, 0.9]])    # states persist
shuffled = np.full((2, 2), 0.5)                  # states reshuffle every step
obs = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]       # a slowly changing sequence
print(log_evidence(obs, A, sticky, D) > log_evidence(obs, A, shuffled, D))
```

## Properties

### 1. Temporal Features

- State Persistence

- [[action_sequences|Action Sequences]]

- Event Prediction

- [[temporal_dependencies|Temporal Dependencies]]

### 2. Processing Characteristics

- [[forward_models|Forward Prediction]]

- Backward Inference

- [[temporal_binding|Temporal Integration]]

- [[sequence_learning|Sequence Learning]]

### 3. Learning Dynamics

- Parameter Adaptation

- [[structure_learning|Structure Learning]]

- Temporal Abstraction

- Sequence Discovery

### 4. Scale Properties

- [[timescale_separation|Timescale Separation]]

- [[hierarchical_processing|Hierarchical Organization]]

- [[information_flow|Information Flow]]

- Temporal Coupling

### 5. Model Selection Properties

- [[model_evidence|Model Evidence]]

- [[complexity_measures|Complexity Measures]]

- Predictive Performance

- [[generalization_capacity|Generalization]]

### 6. Validation Properties

- Temporal Consistency

- Structural Validity

- Behavioral Adequacy

- [[neural_plausibility|Neural Plausibility]]

## Model Selection and Validation

### 1. Selection Criteria

- [[bayesian_model_selection|Bayesian Model Selection]]

- [[information_criteria|Information Criteria]]

- [[cross_validation|Cross Validation]]

- Predictive Validation

### 2. Validation Methods

- Temporal Validation

- Structural Validation

- Empirical Validation

- Theoretical Validation

### 3. Performance Metrics

- [[prediction_error|Prediction Error]]

- [[model_complexity|Model Complexity]]

- Computational Cost

- [[biological_plausibility|Biological Plausibility]]

### 4. Validation Scenarios

- Behavioral Tasks

- Cognitive Tasks

- Motor Tasks

- Learning Tasks

## Practical Considerations

### 1. Implementation Challenges

- Numerical Implementation

- Computational Resources

- Memory Constraints

- Real-time Processing

### 2. Optimization Methods

- Parameter Optimization

- Structure Optimization

- Runtime Optimization

- Memory Optimization

### 3. Debugging Strategies

- Temporal Debugging

- Structural Debugging

- Numerical Debugging

- Performance Debugging

### 4. Documentation Guidelines

- Model Documentation

- Implementation Documentation

- Validation Documentation

- Maintenance Documentation

## Best Practices

### 1. Design Guidelines

1. Choose appropriate timescales

1. Define state transitions

1. Specify observation model

1. Configure learning rules

1. Test temporal consistency

### 2. Common Challenges

- [[long_term_dependencies|Long-term Dependencies]]

- Error Accumulation

- [[state_explosion|State Explosion]]

- [[computational_complexity|Computational Complexity]]

### 3. Optimization Strategies

- Efficient Prediction

- Smart Caching

- [[parallel_processing|Parallel Processing]]

- [[adaptive_timesteps|Adaptive Timesteps]]

## References

1. Friston, K. J., et al. (2017). Active Inference, Curiosity and Insight

1. Parr, T., & Friston, K. J. (2018). The Discrete and Continuous Brain

1. Buckley, C. L., et al. (2017). The free energy principle for action and perception

1. Clark, A. (2013). Whatever next? Predictive brains, situated agents

1. Friston, K. J. (2008). Hierarchical models in the brain

1. Friston, K. J., et al. (2020). Model Selection and Validation in Active Inference

1. Smith, R., et al. (2021). Practical Implementation of Active Inference Models

1. Parr, T., et al. (2022). Temporal Model Comparison in Active Inference

