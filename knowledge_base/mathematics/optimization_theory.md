---
title: "Optimization Theory"
type: "mathematical_concept"
status: stable
---

# Optimization Theory

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

## Overview

Optimization Theory provides the mathematical foundation for finding the best solutions to problems under given constraints. It forms a crucial bridge between theoretical principles and practical implementations in control theory, machine learning, and cognitive systems, particularly in understanding how systems minimize free energy and maximize performance.

## Mathematical Foundation

### Unconstrained Optimization

#### Gradient Descent

```math

x_{k+1} = x_k - \alpha_k \nabla f(x_k)

```

where:

- $x_k$ is current point

- $\alpha_k$ is step size

- $\nabla f$ is gradient

#### Newton's Method

```math

x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1}\nabla f(x_k)

```

where:

- $\nabla^2 f$ is Hessian matrix

### Constrained Optimization

#### Lagrangian

```math

L(x,\lambda) = f(x) + \sum_i \lambda_i g_i(x)

```

where:

- $f(x)$ is objective function

- $g_i(x)$ are constraints

- $\lambda_i$ are Lagrange multipliers

#### KKT Conditions

```math

\begin{align*}

\nabla_x L(x^*,\lambda^*) &= 0 \\

g_i(x^*) &\leq 0 \\

\lambda_i^* g_i(x^*) &= 0 \\

\lambda_i^* &\geq 0

\end{align*}

```

## Implementation

### Gradient-Based Optimizer

```python
import numpy as np

H = np.array([[3.0, 0.5], [0.5, 1.0]])       # illustrative quadratic f(x) = 0.5 x^T H x - b^T x
b = np.array([1.0, 1.0])
grad = lambda x: H @ x - b

x = np.zeros(2)
for _ in range(200):
    x = x - 0.2 * grad(x)                     # x_{k+1} = x_k - alpha grad f(x_k)
print(np.allclose(x, np.linalg.solve(H, b)))
```

### Second-Order Optimizer

```python
x = np.zeros(2)
x = x - np.linalg.solve(H, grad(x))           # Newton step: x - H^{-1} grad f(x)
print(np.allclose(x, np.linalg.solve(H, b)))  # exact in one step for a quadratic
```

### Constrained Optimizer

```python
import numpy as np

# minimise f(x) = x1^2 + x2^2  subject to  g(x) = 1 - x1 - x2 <= 0
# With g active, grad f + lambda grad g = 0 and g = 0 form a linear system.
K = np.array([[2.0, 0.0, -1.0],
              [0.0, 2.0, -1.0],
              [-1.0, -1.0, 0.0]])
rhs = np.array([0.0, 0.0, -1.0])
x1, x2, lam = np.linalg.solve(K, rhs)
g = 1 - x1 - x2
print(round(x1, 3), round(x2, 3), round(lam, 3))
print(np.isclose(g, 0.0), lam >= 0, np.isclose(lam * g, 0.0))   # KKT conditions hold
```

## Applications

### Physical Systems

#### Energy Minimization

- Mechanical equilibrium

- Potential energy

- Variational principles

- Path optimization

#### Control Systems

- Optimal control

- Model predictive control

- Trajectory optimization

- Resource allocation

### Machine Learning

#### Neural Networks

- Loss minimization

- Weight optimization

- Architecture search

- Hyperparameter tuning

#### Probabilistic Models

- Maximum likelihood

- Variational inference

- Expectation maximization

- Bayesian optimization

### Cognitive Systems

#### Active Inference

- Free energy minimization

- Belief updating

- Policy optimization

- Learning and inference

#### Decision Making

- Utility maximization

- Risk minimization

- Multi-objective optimization

- Sequential decision making

## Advanced Topics

### Convex Optimization

- Linear programming

- Quadratic programming

- Semidefinite programming

- Interior point methods

### Stochastic Optimization

- Stochastic gradient descent

- Evolutionary algorithms

- Simulated annealing

- Particle swarm optimization

### Distributed Optimization

- Consensus algorithms

- Decomposition methods

- Federated learning

- Multi-agent optimization

## Best Practices

### Problem Formulation

1. Identify objectives

1. Define constraints

1. Choose variables

1. Select algorithm

### Implementation

1. Initialize properly

1. Handle constraints

1. Monitor convergence

1. Validate solutions

### Validation

1. Optimality conditions

1. Constraint satisfaction

1. Numerical stability

1. Solution quality

## Common Issues

### Technical Challenges

1. Local minima

1. Ill-conditioning

1. Constraint violations

1. Numerical instability

### Solutions

1. Multiple initializations

1. Regularization

1. Constraint relaxation

1. Robust algorithms

## Related Documentation

- [[control_theory]]

- [[statistical_physics]]

- [[machine_learning]]

- [[active_inference]]

- [[variational_inference]]

- [[dynamical_systems]]

## Variational Methods

### Free Energy Optimization

```python
import numpy as np

A = np.array([[0.9, 0.2], [0.1, 0.8]])       # p(o | s), illustrative
prior, o = np.array([0.5, 0.5]), 0
softmax = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()

def free_energy(q):
    return np.sum(q * (np.log(q) - np.log(A[o] * prior)))

theta = np.zeros(2)                            # logits of q(s)
for _ in range(500):
    q = softmax(theta)
    dF_dq = np.log(q) - np.log(A[o] * prior) + 1.0
    theta -= 0.5 * q * (dF_dq - np.sum(q * dF_dq))   # chain rule through the softmax
posterior = A[o] * prior / np.sum(A[o] * prior)
print(np.round(softmax(theta), 3), np.round(posterior, 3))
print(np.isclose(free_energy(softmax(theta)), -np.log(np.sum(A[o] * prior))))   # F reaches -ln p(o)
```

## Gradient Methods

### Natural Gradient Descent

```python
import numpy as np

# Bernoulli q with logit theta, fitted to target p by minimising KL(q || p)
sigmoid = lambda t: 1 / (1 + np.exp(-t))
logit = lambda p: np.log(p / (1 - p))
p_target, theta = 0.8, 0.0                      # illustrative

q = sigmoid(theta)
fisher = q * (1 - q)                            # Fisher information of Bernoulli in logit form
euclid_grad = fisher * (theta - logit(p_target))    # gradient of KL(q || p) w.r.t. theta
natural_grad = euclid_grad / fisher                 # F^{-1} grad
theta_new = theta - natural_grad
print(round(float(sigmoid(theta_new)), 3))      # one natural step reaches the target
```

### Stochastic Optimization

```python
import numpy as np
rng = np.random.default_rng(0)

X = rng.normal(size=(500, 2))
w_true = np.array([2.0, -1.0])
y = X @ w_true + 0.1 * rng.standard_normal(500)

w = np.zeros(2)
for step in range(2000):
    idx = rng.integers(0, 500, size=16)                 # minibatch
    grad = X[idx].T @ (X[idx] @ w - y[idx]) / 16        # stochastic gradient of the squared error
    w -= 0.02 * grad
print(np.round(w, 2))
```

## Documentation Links

- [[docs/research/research_documentation_index|Research Documentation]]

- [[docs/implementation/implementation_guides_index|Implementation Guides]]

- [[docs/api/api_documentation_index|API Documentation]]

- [[docs/examples/usage_examples_index|Usage Examples]]

## References

- bertsekas - Dynamic Programming and Optimal Control

- [[amari]] - Information Geometry and Its Applications

- boyd - Convex Optimization

- sutton - Reinforcement Learning

