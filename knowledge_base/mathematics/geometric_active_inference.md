---
title: Geometric Active Inference
type: mathematical_concept
status: stable
created: 2024-03-20
tags:
- mathematics
- differential_geometry
- information_geometry
- active_inference
semantic_relations:
  - type: foundation
    links:
      - "[[differential_geometry]]"
      - "[[information_geometry]]"
      - "[[symplectic_geometry]]"
      - "[[riemannian_geometry]]"
  - type: implements
    links:
      - "[[active_inference]]"
      - "[[free_energy_principle]]"
  - type: relates
    links:
      - "[[path_integral_free_energy]]"
      - "[[variational_methods]]"
      - "[[optimal_control]]"
---


# Geometric Active Inference

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

## Overview

Geometric Active Inference provides a differential geometric framework for understanding active inference and the free energy principle. This approach reveals deep connections between information geometry, symplectic geometry, and optimal control theory.

## Geometric Structures

### 1. Statistical Manifolds

#### Definition

```math

(\mathcal{M}, g, \nabla^{(\alpha)}, \nabla^{(-\alpha)})

```

where:

- $\mathcal{M}$ is manifold of probability distributions

- $g$ is Fisher-Rao metric

- $\nabla^{(\alpha)}$ are dual connections

#### Fisher-Rao Metric

```math

g_{ij}(\theta) = \int p_\theta(x) \frac{\partial \log p_\theta(x)}{\partial \theta^i} \frac{\partial \log p_\theta(x)}{\partial \theta^j} dx

```

### 2. Belief Space Geometry

#### Tangent Space

```math

T_p\mathcal{M} = \text{span}\left\{\frac{\partial}{\partial \theta^i}\right\}_{i=1}^n

```

#### Cotangent Space

```math

T^*_p\mathcal{M} = \text{span}\{d\theta^i\}_{i=1}^n

```

### 3. Symplectic Structure

#### Canonical Form

```math

\omega = \sum_i dp^i \wedge dq^i

```

where:

- $p^i$ are momenta

- $q^i$ are coordinates

#### Hamiltonian Flow

```math

\dot{q}^i = \frac{\partial H}{\partial p_i}, \quad \dot{p}_i = -\frac{\partial H}{\partial q^i}

```

## Geometric Free Energy

### 1. Free Energy as Action

#### Action Functional

```math

S[q] = \int_0^T \left(g_{ij}\dot{\theta}^i\dot{\theta}^j + F(q_\theta)\right)dt

```

where:

- $F(q_\theta)$ is variational free energy

- $g_{ij}$ is Fisher metric

The metric (kinetic) term is a modelling choice of this page; the free action of the free energy principle integrates the free energy over time without it (see Friston, 2010).

#### Euler-Lagrange Equations

```math

\frac{d}{dt}\frac{\partial L}{\partial \dot{\theta}^i} - \frac{\partial L}{\partial \theta^i} = 0

```

### 2. Natural Gradient Flow

#### Gradient Flow

```math

\dot{\theta}^i = -g^{ij}\frac{\partial F}{\partial \theta^j}

```

#### Parallel Transport

```math

\nabla_{\dot{\gamma}}\dot{\gamma} = 0

```

## Geometric Policy Selection

### 1. Policy Manifold

#### Structure

```math

\mathcal{P} = \{P_\pi : \pi \in \Pi\}

```

where:

- $P_\pi$ is policy distribution

- $\Pi$ is policy space

#### Metric

```math

h_{ij}(\pi) = \mathbb{E}_{P_\pi}\left[\frac{\partial \log P_\pi}{\partial \pi^i}\frac{\partial \log P_\pi}{\partial \pi^j}\right]

```

### 2. Expected Free Energy

#### Geometric Form

```math

G(\pi) = \int_{\mathcal{M}} g_{ij}(\theta)\dot{\theta}^i\dot{\theta}^j d\mu(\theta)

```

This is a schematic path-energy functional specific to this page, not the standard definition of expected free energy (Parr et al., 2022).

#### Policy Update

```math

\dot{\pi}^i = -h^{ij}\frac{\partial G}{\partial \pi^j}

```

## Implementation

### 1. Geometric Integration

```python
import numpy as np

def softmax(theta):
    e = np.exp(theta - theta.max())
    return e / e.sum()

def fisher(theta):
    """g_ij = E_p[d_i log p * d_j log p] for a categorical with softmax parameters."""
    p = softmax(theta)
    score = np.eye(len(p)) - p              # row k: d log p_k / d theta
    return sum(p[k] * np.outer(score[k], score[k]) for k in range(len(p)))

theta = np.array([0.5, -0.2, 0.1])          # illustrative parameters
p = softmax(theta)
print(np.allclose(fisher(theta), np.diag(p) - np.outer(p, p)))   # closed form
```

### 2. Natural Gradient Methods

```python
import numpy as np

def softmax(theta):
    e = np.exp(theta - theta.max())
    return e / e.sum()

prior = np.array([0.6, 0.3, 0.1])           # illustrative target distribution
F = lambda theta: np.sum(softmax(theta) * (np.log(softmax(theta)) - np.log(prior)))

def grad(theta):                             # Euclidean gradient of F = KL(q || prior)
    q = softmax(theta)
    d = np.log(q) - np.log(prior)
    return q * (d - q @ d)

def natural_grad(theta):                     # pseudo-inverse of the Fisher metric diag(q) - q q^T
    q = softmax(theta)
    G = np.diag(q) - np.outer(q, q)
    return np.linalg.pinv(G) @ grad(theta)

for name, direction in [("euclidean", grad), ("natural", natural_grad)]:
    theta = np.array([-1.0, 0.5, 1.5])
    for _ in range(50):
        theta = theta - 0.5 * direction(theta)
    print(name, "F after 50 steps:", F(theta))
```

### 3. Geometric Policy Optimization

```python
import numpy as np

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

G = np.array([2.0, 1.0, 1.5, 3.0])           # illustrative expected free energy per policy
logits = np.zeros(4)
for _ in range(100):
    P = softmax(logits)
    # natural gradient of J = E_P[G] - H[P] w.r.t. the logits is G + log P (up to a constant)
    logits = logits - 0.2 * (G + np.log(P))
print(np.allclose(softmax(logits), softmax(-G), atol=1e-6))
```

## Applications

### 1. Geometric Control

- Optimal transport

- Path planning

- Trajectory optimization

- Feedback control

### 2. Information Processing

- Belief propagation

- Message passing

- Information geometry

- Statistical inference

### 3. Learning Theory

- Natural gradient descent

- Information bottleneck

- Geometric deep learning

- Manifold learning

## Best Practices

### 1. Geometric Methods

1. Preserve invariants

1. Use natural coordinates

1. Implement symplectic integrators

1. Handle parallel transport

### 2. Numerical Stability

1. Monitor geodesic distance

1. Check metric positivity

1. Regularize curvature

1. Control step size

### 3. Implementation

1. Efficient tensor operations

1. Adaptive discretization

1. Geometric integration

1. Parallel computation

## Common Issues

### 1. Technical Challenges

1. Coordinate singularities

1. Metric degeneracy

1. Geodesic completeness

1. Computational complexity

### 2. Solutions

1. Multiple charts

1. Regularization

1. Adaptive methods

1. Efficient algorithms

## Complete Geometric Framework

### Advanced Theoretical Foundation

**Definition** (Statistical Manifold with Active Inference Structure): A statistical manifold $(M, g, \nabla)$ equipped with:

- Riemannian metric $g$ (Fisher information)

- Affine connection $\nabla$

- Free energy functional $F: M \to \mathbb{R}$

- Action functional $S: \text{Path}(M) \to \mathbb{R}$

### Advanced Applications

#### 1. Geometric Deep Learning

- Graph neural networks on statistical manifolds

- Equivariant architectures for active inference

- Manifold-aware optimization

#### 2. Quantum Active Inference

- Geometric phases in quantum systems

- Information geometry of quantum states

- Quantum optimal transport

#### 3. Continuum Mechanics

- Field theories of active inference

- Geometric mechanics of continuous systems

- Variational formulations

## Related Topics

- [[differential_geometry]]

- [[information_geometry]]

- [[symplectic_geometry]]

- [[optimal_control]]

- [[path_integral_free_energy]]

- [[variational_methods]]


## References

- Amari, S., & Nagaoka, H. (2000). Methods of Information Geometry. American Mathematical Society and Oxford University Press.
- Amari, S. (1998). Natural gradient works efficiently in learning. Neural Computation, 10(2), 251-276.
- Arnold, V. I. (1989). Mathematical Methods of Classical Mechanics (2nd ed.). Springer.
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior. MIT Press.
- Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127-138.
