---
external_mappings:
  - target: http://www.wikidata.org/entity/Q188444
    relation: exactMatch
    evidence: https://www.wikidata.org/wiki/Q188444
    rationale: >-
      The source covers manifolds, metrics, connections, curvature and differential
      forms, then applies them to cognitive models. The target is the mathematical
      field studying functions and geometric structures on differentiable manifolds.
      Both denote differential geometry, not only one application or its Riemannian
      subfield.
    reviewed_on: "2026-10-20"
type: mathematical_concept
id: differential_geometry_001
created: 2024-02-06
modified: 2024-02-06
tags:
- mathematics
- differential-geometry
- manifolds
- connections
aliases:
- riemannian-geometry
- geometric-mechanics
- differential_forms
- riemannian_geometry
- riemannian_manifolds
semantic_relations:
  - type: implements
    links:
      - "[[docs/research/research_documentation_index|Research Documentation]]"
      - "[[information_geometry]]"
  - type: uses
    links:
      - "[[tensor_calculus]]"
      - "[[lie_theory]]"
  - type: documented_by
    links:
      - "[[docs/implementation/implementation_guides_index|Implementation Guides]]"
      - "[[docs/api/api_documentation_index|API Documentation]]"
---


# Differential Geometry in Cognitive Modeling

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].


## Overview

Differential geometry provides the mathematical foundation for understanding the geometric structure of state spaces and belief manifolds in cognitive modeling. This document explores differential geometric concepts and their applications in active inference.

## Manifold Theory

### Riemannian Metrics

```python
import numpy as np

# Round sphere in coordinates (theta, phi): ds^2 = dtheta^2 + sin^2(theta) dphi^2
def metric(x):
    return np.array([[1.0, 0.0], [0.0, np.sin(x[0]) ** 2]])

def curve_length(curve, ts):
    """Length of a curve x(t) as the sum of sqrt(g(dx, dx)) over small steps."""
    xs = np.array([curve(t) for t in ts])
    dx = np.diff(xs, axis=0)
    mid = 0.5 * (xs[1:] + xs[:-1])
    return sum(np.sqrt(d @ metric(m) @ d) for d, m in zip(dx, mid))

ts = np.linspace(0, 1, 2001)
equator = lambda t: np.array([np.pi / 2, 2 * np.pi * t])
print(curve_length(equator, ts), 2 * np.pi)           # great circle
```

## Connections and Transport

### Levi-Civita Connection

```python
import numpy as np

def metric(x):
    return np.array([[1.0, 0.0], [0.0, np.sin(x[0]) ** 2]])

def christoffel(x, h=1e-5):
    """Gamma^k_ij = 1/2 g^{kl} (d_i g_lj + d_j g_li - d_l g_ij), by central differences."""
    dg = np.array([(metric(x + h * e) - metric(x - h * e)) / (2 * h) for e in np.eye(2)])  # dg[i, l, j]
    ginv = np.linalg.inv(metric(x))
    return 0.5 * np.einsum("kl,ilj->kij", ginv, dg) \
         + 0.5 * np.einsum("kl,jli->kij", ginv, dg) \
         - 0.5 * np.einsum("kl,lij->kij", ginv, dg)

x = np.array([1.0, 0.3])                      # illustrative point
G = christoffel(x)
print(G[0, 1, 1], -np.sin(x[0]) * np.cos(x[0]))   # Gamma^theta_{phi phi}
print(G[1, 0, 1], 1 / np.tan(x[0]))               # Gamma^phi_{theta phi}
```

### Geodesic Flow

```python
import numpy as np

def geodesic_rhs(state):
    """x'' = -Gamma^k_ij x'^i x'^j on the round sphere, state = (theta, phi, theta', phi')."""
    th, ph, dth, dph = state
    return np.array([dth, dph,
                     np.sin(th) * np.cos(th) * dph ** 2,
                     -2.0 * dth * dph / np.tan(th)])

def rk4(f, y, h, n):
    for _ in range(n):
        k1 = f(y); k2 = f(y + 0.5 * h * k1); k3 = f(y + 0.5 * h * k2); k4 = f(y + h * k3)
        y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y

y0 = np.array([1.0, 0.0, 0.3, 0.8])           # illustrative initial point and velocity
y1 = rk4(geodesic_rhs, y0, 1e-3, 5000)
speed2 = lambda y: y[2] ** 2 + np.sin(y[0]) ** 2 * y[3] ** 2
print("speed^2 conserved:", speed2(y0), speed2(y1))
```

## Curvature Theory

### Riemann Curvature

```python
import numpy as np

def gamma(x):
    """Christoffel symbols of the round sphere, indexed [k, i, j]."""
    G = np.zeros((2, 2, 2))
    G[0, 1, 1] = -np.sin(x[0]) * np.cos(x[0])
    G[1, 0, 1] = G[1, 1, 0] = 1 / np.tan(x[0])
    return G

def riemann(x, h=1e-5):
    """R^r_{s m n} = d_m Gamma^r_{ns} - d_n Gamma^r_{ms} + Gamma^r_{ml} Gamma^l_{ns} - Gamma^r_{nl} Gamma^l_{ms}."""
    dG = np.array([(gamma(x + h * e) - gamma(x - h * e)) / (2 * h) for e in np.eye(2)])   # dG[m, r, n, s]
    G = gamma(x)
    return (np.einsum("mrns->rsmn", dG) - np.einsum("nrms->rsmn", dG)
            + np.einsum("rml,lns->rsmn", G, G) - np.einsum("rnl,lms->rsmn", G, G))

x = np.array([1.0, 0.3])                      # illustrative point
R = riemann(x)
K = R[0, 1, 0, 1] / np.sin(x[0]) ** 2         # sectional curvature = R^theta_{phi theta phi} / g_phiphi
print("Gaussian curvature of the unit sphere:", K)
```

## Lie Theory

### Lie Groups

```python
import numpy as np

def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def exp_so3(w):
    """Rodrigues' formula: exponential map from the Lie algebra so(3) to the group SO(3)."""
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    K = hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

R = exp_so3(np.array([0.3, -0.5, 0.8]))       # illustrative algebra element
print(np.allclose(R.T @ R, np.eye(3)), np.isclose(np.linalg.det(R), 1.0))
```

## Applications to Active Inference

### Belief Manifolds

```python
import numpy as np

def fisher_rao_distance(p, q):
    """Geodesic distance on the probability simplex: x -> 2 sqrt(x) maps it to a sphere of radius 2."""
    return 2.0 * np.arccos(np.clip(np.sum(np.sqrt(p * q)), -1.0, 1.0))

p = np.array([0.7, 0.2, 0.1])                 # illustrative beliefs
q = np.array([0.1, 0.3, 0.6])
print(fisher_rao_distance(p, q))
```

## Documentation Links

- [[docs/research/research_documentation_index|Research Documentation]]

- [[docs/implementation/implementation_guides_index|Implementation Guides]]

- [[docs/api/api_documentation_index|API Documentation]]

- [[docs/examples/usage_examples_index|Usage Examples]]

## References

- do carmo - Riemannian Geometry

- lee - Introduction to Smooth Manifolds

- kobayashi nomizu - Foundations of Differential Geometry

- marsden ratiu - Introduction to Mechanics and Symmetry

---

title: Differential Geometry

type: mathematical_concept

status: stable

created: 2024-02-12

tags:

  - mathematics

  - geometry

  - topology

semantic_relations:

  - type: foundation

    links:

      - manifold theory

      - [[topology]]

  - type: relates

    links:

      - [[information_geometry]]

      - [[lie_groups]]

      - tensor analysis

---

# Differential Geometry

## Core Concepts

### Manifolds

1. **Smooth Manifolds**

   ```math

   M = \{(U_α,φ_α) | α ∈ A\}

   ```

   where:

   - U_α are open sets

   - φ_α are coordinate charts

   - A is atlas index set

1. **Tangent Space**

   ```math

   T_pM = span\{\frac{∂}{∂x^i}|_p\}

   ```

   where:

   - p is point on manifold

   - x^i are local coordinates

### Differential Forms

1. **Exterior Derivative**

   ```math

   dω = \sum_{i_1<...<i_k} \frac{∂ω_{i_1...i_k}}{∂x^j}dx^j ∧ dx^{i_1} ∧ ... ∧ dx^{i_k}

   ```

   where:

   - ω is k-form

   - ∧ is wedge product

1. **Integration**

   ```math

   \int_M ω = \sum_α \int_{φ_α(U_α)} (φ_α^{-1})^*ω

   ```

   where:

   - ω is n-form

   - φ_α are charts

### Riemannian Geometry

1. **Metric Tensor**

   ```math

   ds² = g_{ij}dx^idx^j

   ```

   where:

   - g_{ij} is metric components

   - dx^i are coordinate differentials

1. **Christoffel Symbols**

   ```math

   Γ^k_{ij} = \frac{1}{2}g^{kl}(∂_ig_{jl} + ∂_jg_{il} - ∂_lg_{ij})

   ```

   where:

   - g^{kl} is inverse metric

   - ∂_i is partial derivative

## Advanced Concepts

### Connection Theory

1. **Covariant Derivative**

   ```math

   ∇_X Y = (X^i∂_iY^k + Γ^k_{ij}X^iY^j)\frac{∂}{∂x^k}

   ```

   where:

   - X,Y are vector fields

   - Γ^k_{ij} are connection coefficients

1. **Parallel Transport**

   ```math

   \frac{D}{dt}V^i + Γ^i_{jk}\frac{dx^j}{dt}V^k = 0

   ```

   where:

   - V^i are vector components

   - D/dt is covariant derivative

### Curvature

1. **Riemann Tensor**

   ```math

   R^i_{jkl} = ∂_kΓ^i_{jl} - ∂_lΓ^i_{jk} + Γ^i_{mk}Γ^m_{jl} - Γ^i_{ml}Γ^m_{jk}

   ```

   where:

   - Γ^i_{jk} are Christoffel symbols

1. **Ricci Tensor**

   ```math

   R_{ij} = R^k_{ikj}

   ```

   where:

   - R^k_{ikj} is Riemann tensor

### Lie Groups

1. **Lie Algebra**

   ```math

   [X,Y] = XY - YX

   ```

   where:

   - X,Y are vector fields

   - [,] is Lie bracket

1. **Exponential Map**

   ```math

   exp: g → G, exp(X) = \sum_{n=0}^∞ \frac{X^n}{n!}

   ```

   where:

   - g is Lie algebra

   - G is Lie group

## Applications

### Information Geometry

1. **Fisher Metric**

   ```math

   g_{ij}(θ) = E[-\frac{∂²}{∂θ^i∂θ^j}log p(x|θ)]

   ```

   where:

   - θ are statistical parameters

   - p(x|θ) is probability model

1. **α-Connections**

   ```math

   Γ^{(α)}_{ijk} = E[\frac{∂}{\∂θ^i}log p(x|θ)\frac{∂}{\∂θ^j}log p(x|θ)\frac{∂}{\∂θ^k}log p(x|θ)]

   ```

   where:

   - α is connection parameter

### General Relativity

1. **Einstein Field Equations**

   ```math

   R_{μν} - \frac{1}{2}Rg_{μν} + Λg_{μν} = \frac{8πG}{c^4}T_{μν}

   ```

   where:

   - R_{μν} is Ricci tensor

   - T_{μν} is stress-energy tensor

1. **Geodesic Equation**

   ```math

   \frac{d²x^μ}{dτ²} + Γ^μ_{αβ}\frac{dx^α}{dτ}\frac{dx^β}{dτ} = 0

   ```

   where:

   - τ is proper time

   - Γ^μ_{αβ} are Christoffel symbols

### Machine Learning

1. **Natural Gradient**

   ```math

   θ_{t+1} = θ_t - ηg^{ij}(θ_t)∂_jL(θ_t)

   ```

   where:

   - g^{ij} is inverse Fisher metric

   - L is loss function

1. **Manifold Learning**

   ```math

   min_f \int_M ||∇f||²_g dV_g

   ```

   where:

   - f is embedding

   - g is metric

   - dV_g is volume form

## Advanced Topics

### Symplectic Geometry

1. **Symplectic Form**

   ```math

   ω = dx^i ∧ dp_i

   ```

   where:

   - x^i are position coordinates

   - p_i are momentum coordinates

1. **Hamiltonian Flow**

   ```math

   \dot{x}^i = \frac{∂H}{∂p_i}, \dot{p}_i = -\frac{∂H}{∂x^i}

   ```

   where:

   - H is Hamiltonian

   - x^i,p_i are canonical coordinates

### Complex Geometry

1. **Kähler Manifolds**

   ```math

   ω = ig_{αβ̄}dz^α ∧ dz̄^β

   ```

   where:

   - g_{αβ̄} is Hermitian metric

   - z^α are complex coordinates

1. **Holomorphic Forms**

   ```math

   ∂ω = 0

   ```

   where:

   - ∂ is Dolbeault operator

   - ω is differential form

## Future Directions

### Emerging Areas

1. **Geometric Deep Learning**

   - Group equivariant networks

   - Manifold optimization

   - Topological data analysis

1. **Quantum Geometry**

   - Non-commutative geometry

   - Quantum gravity

   - String theory

### Open Problems

1. **Theoretical Challenges**

   - Geometric flows

   - Singular geometries

   - Mirror symmetry

1. **Practical Challenges**

   - Numerical methods

   - High-dimensional manifolds

   - Discrete geometry

## Related Topics

1. [[topology|Topology]]

1. [[lie_theory|Lie Theory]]

1. [[algebraic_geometry|Algebraic Geometry]]

1. Geometric Analysis

