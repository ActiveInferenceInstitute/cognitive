---
title: Active Inference Implementation Guide
type: implementation
status: stable
created: 2025-01-01
updated: 2025-01-01
tags:
  - active_inference
  - implementation
  - python
  - practical_guide
semantic_relations:
  - type: implements
    links:
      - "[[mathematics/free_energy_principle]]"
      - "[[cognitive/active_inference]]"
---

# Active Inference Implementation Guide

This document provides practical implementations of active inference principles, offering code examples, algorithmic details, and step-by-step guides for applying active inference to real-world problems. The implementations build on the mathematical foundations while providing actionable code for researchers and developers.

## 🐍 Python Implementation Framework

### Core Active Inference Components

#> [!note] Illustrative pseudocode
> The code in this section is an illustrative numpy implementation
> sketch; it is not part of this repository's `cognitive`
> package. For the installed implementation, see the canonical example in
> the root `README.md`.

### Generative Model Definition

```python
import numpy as np

n_states = 3                                  # illustrative sizes
A = np.full((n_states, n_states), 0.05)       # likelihood p(o | s)
np.fill_diagonal(A, 0.90)
B = np.stack([np.eye(n_states),               # action 0: stay
              np.roll(np.eye(n_states), 1, axis=0)])   # action 1: move to next state
C = np.array([0.0, 1.0, 2.0])                 # log preferences over outcomes
D = np.ones(n_states) / n_states              # prior over initial states

assert np.allclose(A.sum(axis=0), 1) and np.allclose(B.sum(axis=1), 1)
```

#### Variational Free Energy Computation

```python
import numpy as np

A = np.array([[0.90, 0.05, 0.05],
              [0.05, 0.90, 0.05],
              [0.05, 0.05, 0.90]])            # p(o | s), illustrative
prior = np.array([0.5, 0.3, 0.2])
o = 2                                         # observed outcome

def free_energy(q):
    return np.sum(q * (np.log(q) - np.log(prior) - np.log(A[o])))

q = prior * A[o] / (prior @ A[o])             # the minimiser of F is the exact posterior
print(q.round(3), free_energy(q), -np.log(prior @ A[o]))   # F equals surprise there
print(free_energy(np.ones(3) / 3))            # any other q has higher F
```

#### Expected Free Energy for Action Selection

```python
import numpy as np

A = np.full((3, 3), 0.05)
np.fill_diagonal(A, 0.90)                     # p(o | s), illustrative
B = np.stack([np.eye(3), np.roll(np.eye(3), 1, axis=0)])   # stay / move
log_C = np.array([0.0, 1.0, 2.0])             # log preferences
log_C = log_C - np.log(np.exp(log_C).sum())   # normalise to ln p(o)

def expected_free_energy(action, qs):
    qs_next = B[action] @ qs
    qo = A @ qs_next
    risk = np.sum(qo * (np.log(qo) - log_C))
    ambiguity = qs_next @ -(A * np.log(A)).sum(axis=0)
    return risk + ambiguity

qs = np.array([0.1, 0.8, 0.1])
G = [expected_free_energy(a, qs) for a in range(2)]
print(np.round(G, 3), int(np.argmin(G)))      # moving toward the preferred state wins
```

### Complete Active Inference Agent

```python
import numpy as np

rng = np.random.default_rng(3)
A = np.full((3, 3), 0.05)
np.fill_diagonal(A, 0.90)                     # p(o | s), illustrative
B = np.stack([np.eye(3), np.roll(np.eye(3), 1, axis=0)])   # stay / move
C = np.array([0.0, 1.0, 2.0])
log_C = C - np.log(np.exp(C).sum())
ambiguity = -(A * np.log(A)).sum(axis=0)

def efe(a, qs):
    qs_next = B[a] @ qs
    qo = A @ qs_next
    return np.sum(qo * (np.log(qo) - log_C)) + qs_next @ ambiguity

s, qs = 0, np.ones(3) / 3                     # true state, belief
for t in range(6):
    o = rng.choice(3, p=A[:, s])              # environment emits an outcome
    qs = qs * A[o]; qs /= qs.sum()            # perceive
    a = int(np.argmin([efe(a, qs) for a in range(2)]))   # act
    s = int(np.argmax(B[a][:, s]))            # environment transition
    qs = B[a] @ qs                            # predict the next state
    print(t, "obs", o, "action", a, "belief", qs.round(2))
```


## 🔧 Implementation Tools and Libraries

### Core Dependencies

See `docs/examples/README.md` for runnable examples.


## 📚 References and Extensions

### Core Implementation References

- [[mathematics/free_energy_principle|Free Energy Principle]]
- [[cognitive/active_inference|Active Inference Theory]]
- Existing Implementation Tools

### Advanced Topics

- **Hierarchical Active Inference**: Multi-timescale inference
- **Multi-Agent Systems**: Collective active inference
- **Continuous State Spaces**: Differential equation formulations
- **Quantum Active Inference**: Quantum probability extensions

### Performance Optimization

- **GPU Acceleration**: CUDA implementations for large-scale models
- **Distributed Computing**: Parallel inference across multiple machines
- **Approximate Inference**: Variational approximations for complex models
- **Online Learning**: Continuous model updating during deployment

---

> **Practical Implementation**: Provides working code examples for active inference applications.

---

> **Modular Design**: Components can be mixed and matched for different use cases.

---

> **Extensible Framework**: Easy to extend for hierarchical and multi-agent scenarios.

---

> **Related**: [[knowledge_base/agents/README|Swarm Intelligence Implementation]] - Collective Active Inference in real systems.

---

> **Research Enablement**: Accelerates implementation of active inference in real systems.

## References

- Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior. MIT Press.
- Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: A process theory. Neural Computation, 29(1), 1-49.
- Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., & Friston, K. (2020). Active inference on discrete state-spaces: A synthesis. Journal of Mathematical Psychology, 99, 102447.
- Smith, R., Friston, K. J., & Whyte, C. J. (2022). A step-by-step tutorial on active inference and its application to empirical data. Journal of Mathematical Psychology, 107, 102632.
- Heins, C., Millidge, B., Demekas, D., Klein, B., Friston, K., Couzin, I. D., & Tschantz, A. (2022). pymdp: A Python library for active inference in discrete state spaces. Journal of Open Source Software, 7(73), 4098.
- Sajid, N., Ball, P. J., Parr, T., & Friston, K. J. (2021). Active inference: Demystified and compared. Neural Computation, 33(3), 674-712.
