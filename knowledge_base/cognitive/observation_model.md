---
title: Observation Model
type: concept
status: stable
created: 2026-02-06
updated: 2026-02-07
tags:
  - observation-model
  - A-matrix
  - likelihood
  - generative-model
  - active-inference
semantic_relations:
  - type: relates
    links:
      - [[generative_model]]
      - [[matrix_specifications]]
      - [[transition_model]]
      - [[perceptual_inference]]
---

# Observation Model

## Overview

The observation model (A matrix) specifies the likelihood mapping from hidden states to observations: $p(o|s)$. It is the core component that connects an agent's internal model of the world to sensory evidence. In Active Inference, the A matrix mediates perceptual inference by constraining which observations are expected given different state hypotheses.

## Mathematical Specification

### Discrete Case

```math
A_{ij} = p(o = i | s = j) \quad \text{where } \sum_i A_{ij} = 1 \; \forall j
```

### Multi-Modal Observations

For factored observation spaces with modalities $m$:

```math
p(o_1, o_2, ..., o_M | s) = \prod_{m=1}^M A^{(m)}_{i_m, j} = \prod_m p(o_m | s)
```

### Multi-Factor States

When states are factored across $F$ factors:

```math
A^{(m)}_{i, j_1, j_2, ..., j_F} = p(o_m = i | s_1 = j_1, s_2 = j_2, ...)
```

## Implementation

```python
import numpy as np

def build_observation_model(n_obs, n_states, mapping='identity', noise=0.0):
    if mapping == 'identity':
        A = np.eye(n_obs, n_states)
    elif mapping == 'noisy_identity':
        A = (1 - noise) * np.eye(n_obs, n_states) + noise / n_obs
    elif mapping == 'random':
        A = np.random.dirichlet(np.ones(n_obs), size=n_states).T
    else:
        raise ValueError(f"Unknown mapping type: {mapping}")
    
    # Each column should be a valid probability distribution
    A = A / A.sum(axis=0, keepdims=True)
    return A

# T-Maze example: location modality
A_location = np.array([
    [1, 0, 0, 0],  # observe "center" in center
    [0, 1, 0, 0],  # observe "left" in left arm
    [0, 0, 1, 0],  # observe "right" in right arm
    [0, 0, 0, 1],  # observe "cue" in cue location
])
```

### Learning the A Matrix

The A matrix is learned via Dirichlet updates:

```math
a_{ij}^{(t+1)} = a_{ij}^{(t)} + \eta_A \cdot o_i^{(t)} \cdot s_j^{(t)} \quad \Rightarrow \quad A_{ij} = \frac{a_{ij}}{\sum_k a_{kj}}
```

where $a$ is the Dirichlet concentration parameter (count matrix) and $\eta_A$ is the learning rate.

## Information-Theoretic Properties

The A matrix determines key information quantities:

| Property | Formula | Meaning |
| --- | --- | --- |
| Ambiguity | $H[p(o\|s)] = -\sum_i A_{ij} \ln A_{ij}$ | Observation noise per state |
| Mutual information | $I(o; s) = H(o) - H(o\|s)$ | State discriminability |
| Determinism | $\max_i A_{ij}$ | Observation confidence |

## Related Topics

- [[generative_model]] — Generative model overview
- [[matrix_specifications]] — Full matrix specifications
- [[transition_model]] — Transition model (B matrix)
- [[perceptual_inference]] — Perception through A
- [[learning_mechanisms]] — Learning models and A updates
