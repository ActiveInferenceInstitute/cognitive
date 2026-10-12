---
title: Active Inference Agent Architecture
type: implementation
status: stable
created: 2024-02-07
updated: 2026-01-03
tags:
  - active_inference
  - agent_architecture
  - implementation
  - cognitive_modeling
semantic_relations:
  - type: implements
    links:
      - "[[active_inference]]"
      - "[[knowledge_base/mathematics/active_inference_theory]]"
  - type: foundation
    links:
      - "[[knowledge_base/mathematics/free_energy_principle]]"
      - "[[generative_model]]"
      - "[[belief_updating]]"
  - type: documented_by
    links:
      - "[[knowledge_base/active_inference_implementation]]"
      - "[[knowledge_base/agents/GenericPOMDP/README]]"
      - "[[knowledge_base/agents/Continuous_Time/README]]"
aliases:
  - active inference agents
  - active_inference_agents
---

# Implementing Active Inference Agents

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

## Overview

This guide provides a comprehensive approach to implementing active inference agents, from basic principles to advanced features. We'll cover the theoretical foundations, mathematical implementations, and practical considerations.

## Core Components

### 1. Generative Model

```python
import numpy as np
from cognitive.models.active_inference import DiscreteGenerativeModel

A = np.array([[0.9, 0.2], [0.1, 0.8]])                    # p(o | s), illustrative
B = np.stack([np.eye(2), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2)  # p(s' | s, a)
C = np.array([2.0, 0.0])                                  # log preferences over outcomes
D = np.array([0.5, 0.5])                                  # prior over states
E = np.array([0.5, 0.5])                                  # prior over actions
model = DiscreteGenerativeModel(A=A, B=B, C=C, D=D, E=E)
```

### 2. Variational Inference

```python
beliefs = model.posterior(0, model.D)        # p(s | o = 0) by Bayes' rule
print(np.round(beliefs, 3))
```

### 3. Policy Selection

```python
from cognitive.utils.matrix_utils import softmax

G = np.array([model.expected_free_energy(beliefs, a)[0] for a in range(model.num_actions)])
action_probs = softmax(-G)                   # lower expected free energy, higher probability
print(np.round(G, 3), np.round(action_probs, 3))

# Multi-step policies are scored with the discounted sum of per-step expected free energy
policies = model.enumerate_policies(horizon=2)
scores = [model.evaluate_policy(beliefs, pi, discount_factor=0.9) for pi in policies]
print(policies[int(np.argmin(scores))])
```

## Implementation Steps

### 1. Main Agent Class

```python
rng = np.random.default_rng(0)
true_state, beliefs = 0, model.D.copy()

for t in range(5):
    o = rng.choice(2, p=A[:, true_state])                       # environment emits an observation
    beliefs = model.posterior(o, beliefs)                       # perceive: update beliefs
    G = [model.expected_free_energy(beliefs, a)[0] for a in range(2)]
    action = int(np.argmin(G))                                  # act: minimise expected free energy
    true_state = rng.choice(2, p=B[:, true_state, action])      # environment transitions
    beliefs = model.predict_states(beliefs, action)             # prior for the next step
    print(t, o, action, np.round(beliefs, 2))
```

## Configuration Examples

### Basic Configuration

```yaml

agent_config:

  state_dim: 10

  obs_dim: 5

  action_dim: 3

  learning:

    learning_rate: 0.1

    temperature: 1.0

  model:

    hidden_dims: [64, 32]

    activation: "relu"

```

### Hierarchical Configuration

```yaml

hierarchical_config:

  levels:

    - state_dim: 20

      temporal_scale: 1

    - state_dim: 10

      temporal_scale: 5

    - state_dim: 5

      temporal_scale: 10

```

## Common Issues and Solutions

### 1. Numerical Stability

```python
import numpy as np
from cognitive.utils.matrix_utils import softmax

logits = np.array([1000.0, 1001.0, 1002.0])      # np.exp(logits) alone overflows
print(softmax(logits))                           # the maximum is subtracted before exponentiating
```

### 2. Belief Normalization

```python
from cognitive.utils.matrix_utils import normalize_mass

print(normalize_mass(np.array([1e-300, 3e-300])))   # scale-invariant, tiny entries are not treated as empty
print(normalize_mass(np.zeros(3)))                  # an all-zero slice becomes uniform
```

## References

- [[active_inference|Active Inference Theory]]

- [[knowledge_base/mathematics/free_energy_theory|Free Energy Theory]]

- [[knowledge_base/mathematics/variational_methods|Variational Methods]]

- Basic Example

