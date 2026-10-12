---
title: Active Inference Theory
type: mathematical_concept
id: active_inference_theory_001
created: 2024-02-06
modified: 2025-12-18
tags:
- mathematics
- active-inference
- free-energy
- cognitive-modeling
aliases:
- active-inference-framework
- free-energy-principle
semantic_relations:
  - type: implements
    links:
      - "[[free_energy_principle]]"
      - "[[knowledge_base/cognitive/active_inference]]"
  - type: foundation
    links:
      - "[[variational_methods]]"
      - "[[information_theory]]"
      - "[[optimal_control]]"
      - "[[bayesian_inference]]"
  - type: documented_by
    links:
      - "[[knowledge_base/active_inference_implementation]]"
      - "[[knowledge_base/active_inference_integration]]"
      - "[[geometric_active_inference]]"
      - "[[measure_theoretic_active_inference]]"
      - "[[stochastic_active_inference]]"
  - type: extends
    links:
      - "[[docs/implementation/implementation_guides_index|Implementation Guides]]"
      - "[[docs/api/api_documentation_index|API Documentation]]"
---

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

## Overview

Active Inference unifies perception, learning, and action under the free energy principle. This document provides a comprehensive mathematical treatment of Active Inference and its applications in cognitive modeling.

## Core Mathematics

### Generative Model

```python
import numpy as np

# Discrete generative model: A[o, s] = P(o | s), B[s', s, a] = P(s' | s, a), C = log-preferences, D = prior
A = np.array([[0.9, 0.2], [0.1, 0.8]])        # illustrative values
B = np.stack([np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]])], axis=2)   # action 0 stays, action 1 swaps
C = np.array([2.0, 0.0])                       # prefers observation 0
D = np.array([0.5, 0.5])

assert np.allclose(A.sum(axis=0), 1) and np.allclose(B.sum(axis=0), 1)
rng = np.random.default_rng(0)
s = rng.choice(2, p=D)                         # hidden state
o = rng.choice(2, p=A[:, s])                   # observation generated from it
print("state", s, "observation", o)
```

### Variational Inference

```python
import numpy as np

A = np.array([[0.9, 0.2], [0.1, 0.8]])        # illustrative likelihood P(o | s)
D = np.array([0.5, 0.5])                       # prior over states
o = 0                                          # observed outcome

def free_energy(q):
    """F = KL(q || D) - E_q[ln P(o | s)]  (complexity minus accuracy)."""
    return np.sum(q * (np.log(q) - np.log(D))) - np.sum(q * np.log(A[o]))

posterior = A[o] * D / np.sum(A[o] * D)       # exact posterior P(s | o)
log_evidence = np.log(np.sum(A[o] * D))
print(free_energy(posterior), -log_evidence)   # equal at the exact posterior
print(free_energy(np.array([0.5, 0.5])) > -log_evidence)   # any other q has higher F
```

### Expected Free Energy

```python
import numpy as np
from cognitive.utils.matrix_utils import expected_free_energy

A = np.array([[0.9, 0.2], [0.1, 0.8]])        # illustrative model
B = np.stack([np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]])], axis=2)
C = np.array([2.0, 0.0])
beliefs = np.array([0.2, 0.8])

G = [expected_free_energy(A, B, C, beliefs, action=a) for a in range(2)]
print("expected free energy per action:", G)     # risk + ambiguity; lower is better
```

## Policy Selection

### Action Selection

```python
import numpy as np
from cognitive.utils.matrix_utils import expected_free_energy, softmax

A = np.array([[0.9, 0.2], [0.1, 0.8]])        # illustrative model
B = np.stack([np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]])], axis=2)
C = np.array([2.0, 0.0])
beliefs = np.array([0.2, 0.8])
gamma = 4.0                                    # illustrative precision

G = np.array([expected_free_energy(A, B, C, beliefs, a) for a in range(2)])
policy_posterior = softmax(-gamma * G)         # P(action) is proportional to exp(-gamma * G)
action = np.random.default_rng(0).choice(2, p=policy_posterior)
print(policy_posterior, action)
```

## Learning

### Parameter Learning

```python
import numpy as np

# Dirichlet concentration parameters for A; the expected likelihood is the column-normalised counts
a = np.ones((2, 2))                            # flat prior counts
rng = np.random.default_rng(1)
A_true = np.array([[0.9, 0.2], [0.1, 0.8]])    # illustrative generating process
for _ in range(2000):
    s = rng.integers(2)
    o = rng.choice(2, p=A_true[:, s])
    q_s = np.eye(2)[s]                         # posterior over s (here: s is observed)
    a[o] += q_s                                # a <- a + o (x) q(s)
print(a / a.sum(axis=0))                       # close to A_true
```

## Documentation Links

- [[docs/research/research_documentation_index|Research Documentation]]

- [[docs/implementation/implementation_guides_index|Implementation Guides]]

- [[docs/api/api_documentation_index|API Documentation]]

- [[docs/examples/usage_examples_index|Usage Examples]]

## References

- friston - Active Inference

- parr - Mathematical Foundations

- da costa - Information Geometry

- bogacz - Tutorial Review

