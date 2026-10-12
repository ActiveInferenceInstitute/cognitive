---
type: mathematical_concept
id: optimal_control_001
created: 2024-02-06
modified: 2024-02-06
tags:
- mathematics
- optimal-control
- control-theory
- optimization
aliases:
- control-theory
- dynamic-optimization
- pontryagin_principle
- optimal_control_theory
semantic_relations:
  - type: implements
    links:
      - "[[docs/research/research_documentation_index|Research Documentation]]"
      - "[[active_inference_theory]]"
  - type: uses
    links:
      - "[[variational_calculus]]"
      - "[[dynamic_programming]]"
  - type: documented_by
    links:
      - "[[docs/implementation/implementation_guides_index|Implementation Guides]]"
      - "[[docs/api/api_documentation_index|API Documentation]]"
---


# Optimal Control Theory in Cognitive Modeling

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].


## Overview

Optimal control theory provides the mathematical foundation for understanding action selection and policy optimization in cognitive systems. This document explores control theoretic principles and their applications in active inference.

## Dynamic Systems

### State Space Models

```python
import numpy as np

# Double integrator x = (position, velocity), x_dot = A x + B u, Euler-discretised
dt = 0.05
A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([[0.0], [1.0]])

def step(x, u):
    return x + dt * (A @ x + B @ u)

x = np.array([1.0, 0.0])                      # illustrative initial state
for _ in range(100):
    x = step(x, np.array([-0.5]))             # constant illustrative control
print(x)
```

## Optimal Control

### Hamilton-Jacobi-Bellman

```python
import numpy as np

# Finite-horizon dynamic programming for x_dot = u with running cost x^2 + u^2 (illustrative).
# The HJB equation for the infinite-horizon problem has the solution V(x) = x^2.
dt, steps = 0.05, 400
xs = np.linspace(-2, 2, 201)
us = np.linspace(-2, 2, 81)
V = np.zeros_like(xs)
for _ in range(steps):
    Q = (xs[:, None] ** 2 + us[None, :] ** 2) * dt + np.interp(xs[:, None] + us[None, :] * dt, xs, V)
    V = Q.min(axis=1)
print(V[np.searchsorted(xs, 1.0)])           # close to V(1) = 1
```

## Policy Optimization

### Linear Quadratic Regulator

```python
import numpy as np

dt = 0.05
A = np.array([[1.0, dt], [0.0, 1.0]])         # discrete double integrator
B = np.array([[0.0], [dt]])
Q, R = np.eye(2), np.array([[0.1]])           # illustrative cost weights

def riccati_gain(A, B, Q, R, horizon=500):
    """Backward Riccati recursion; returns the gain K of u = -K x."""
    P = Q.copy()
    for _ in range(horizon):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ K)
    return K

K = riccati_gain(A, B, Q, R)
print("closed-loop spectral radius:", max(abs(np.linalg.eigvals(A - B @ K))))
x = np.array([1.0, 0.0])
for _ in range(200):
    x = (A - B @ K) @ x
print("state after 200 steps:", x)
```

### Model Predictive Control

```python
import numpy as np

dt = 0.05
A = np.array([[1.0, dt], [0.0, 1.0]])
B = np.array([[0.0], [dt]])
Q, R, u_max = np.eye(2), np.array([[0.1]]), 1.0       # illustrative weights and input bound

def first_gain(N):
    """Gain of the first step of an N-step LQR problem."""
    P = Q.copy()
    for _ in range(N):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ K)
    return K

# Dynamics and costs are time-invariant, so the receding-horizon gain is the same at every step;
# only the first input of each horizon is applied, with the bound enforced by clipping.
K = first_gain(20)
x = np.array([2.0, 0.0])
for _ in range(300):
    u = np.clip(-K @ x, -u_max, u_max)
    x = A @ x + B @ u
print("state after 300 steps:", x)
```

## Documentation Links

- [[docs/research/research_documentation_index|Research Documentation]]

- [[docs/implementation/implementation_guides_index|Implementation Guides]]

- [[docs/api/api_documentation_index|API Documentation]]

- [[docs/examples/usage_examples_index|Usage Examples]]

## References

- bertsekas - Dynamic Programming and Optimal Control

- kirk - Optimal Control Theory

- lewis - Optimal Control

- rawlings - Model Predictive Control

