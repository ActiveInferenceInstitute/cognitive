---
title: Geometric Mechanics
description: "Mathematical background on geometric mechanics, covering phase space, symplectic structure, variational principles, momentum maps and reduction, with geometric integrators."
type: mathematical_concept
status: stable
created: 2024-02-12
tags:
- mathematics
- mechanics
- physics
- geometry
semantic_relations:
  - type: foundation
    links:
      - "[[symplectic_geometry]]"
      - "[[lie_theory]]"
  - type: relates
    links:
      - "[[hamiltonian_mechanics]]"
      - "[[lagrangian_mechanics]]"
      - "[[control_theory]]"
---


# Geometric Mechanics

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

## Core Concepts

### Phase Space Structure

1. **Cotangent Bundle**

   ```math

   T^*Q = \{(q,p) : q \in Q, p \in T^*_qQ\}

   ```

   where:

   - Q is configuration space

   - T^*_qQ is cotangent space

1. **Canonical Symplectic Form**

   ```math

   ω = \sum_{i=1}^n dp_i ∧ dq_i

   ```

   where:

   - p_i are momenta

   - q_i are coordinates

### Variational Principles

1. **Hamilton's Action**

   ```math

   S[q] = \int_{t_1}^{t_2} L(q,\dot{q},t)dt

   ```

   where:

   - L is Lagrangian

   - q is path

1. **Euler-Lagrange Equations**

   ```math

   \frac{d}{dt}\frac{∂L}{∂\dot{q}} - \frac{∂L}{∂q} = 0

   ```

   where:

   - L is Lagrangian

   - q is coordinate

## Advanced Concepts

### Momentum Maps

1. **Definition**

   ```math

   J: T^*Q \to \mathfrak{g}^*, \quad ⟨J(α_q),ξ⟩ = ⟨α_q,ξ_Q(q)⟩

   ```

   where:

   - g* is Lie coalgebra

   - ξ_Q is infinitesimal generator

1. **Conservation Law**

   ```math

   \frac{d}{dt}J(z(t)) = 0

   ```

   where:

   - z(t) is solution

   - J is momentum map

### Reduction Theory

1. **Symplectic Reduction**

   ```math

   P_μ = J^{-1}(μ)/G_μ

   ```

   where:

   - μ is momentum value

   - G_μ is isotropy group

1. **Reduced Dynamics**

   ```math

   i_{X_h}ω_μ = dh_μ

   ```

   where:

   - ω_μ is reduced form

   - h_μ is reduced Hamiltonian

## Implementation

### Geometric Integrators

```python
import numpy as np

def verlet(q, p, grad_V, h, n):
    """Stormer-Verlet (leapfrog) for H = p^2/2 + V(q), unit mass."""
    traj = [(q, p)]
    for _ in range(n):
        p = p - 0.5 * h * grad_V(q)
        q = q + h * p
        p = p - 0.5 * h * grad_V(q)
        traj.append((q, p))
    return np.array(traj)

def explicit_euler(q, p, grad_V, h, n):
    traj = [(q, p)]
    for _ in range(n):
        q, p = q + h * p, p - h * grad_V(q)
        traj.append((q, p))
    return np.array(traj)

grad_V = lambda q: q                    # harmonic oscillator, illustrative
H = lambda z: 0.5 * z[:, 1] ** 2 + 0.5 * z[:, 0] ** 2
for name, f in [("verlet", verlet), ("euler", explicit_euler)]:
    z = f(1.0, 0.0, grad_V, 0.1, 1000)
    print(name, "max |H - H0| =", np.abs(H(z) - H(z)[0]).max())
```

### Reduction Computations

```python
import numpy as np

# Central potential V(r) = -1/|r|: rotation symmetry => angular momentum is the momentum map
grad_V = lambda q: q / np.linalg.norm(q) ** 3
J = lambda q, p: q[0] * p[1] - q[1] * p[0]

q, p, h = np.array([1.0, 0.0]), np.array([0.0, 0.9]), 0.01   # illustrative initial state
J0 = J(q, p)
for _ in range(5000):
    p = p - 0.5 * h * grad_V(q)
    q = q + h * p
    p = p - 0.5 * h * grad_V(q)
print("angular momentum drift:", abs(J(q, p) - J0))
```

## Advanced Topics

### Lie-Poisson Systems

1. **Lie-Poisson Bracket**

   ```math

   \{F,G\}_{LP}(μ) = ⟨μ,[∇F,∇G]⟩

   ```

   where:

   - μ is momentum

   - [,] is Lie bracket

1. **Euler-Poincaré Equations**

   ```math

   \frac{d}{dt}\frac{δl}{δξ} = ad^*_ξ\frac{δl}{δξ}

   ```

   where:

   - l is reduced Lagrangian

   - ξ is velocity

### Discrete Mechanics

1. **Discrete Euler-Lagrange**

   ```math

   D_2L_d(q_{k-1},q_k) + D_1L_d(q_k,q_{k+1}) = 0

   ```

   where:

   - L_d is discrete Lagrangian

   - q_k are positions

1. **Discrete Noether**

   ```math

   J_d^+ = J_d^-

   ```

   where:

   - J_d is discrete momentum

   - ± denotes time shift

## Future Directions

### Emerging Areas

1. **Geometric Deep Learning**

   - Structure-Preserving Networks

   - Hamiltonian Neural Networks

   - Symplectic Networks

1. **Applications**

   - Quantum Computing

   - Molecular Dynamics

   - Plasma Physics

### Open Problems

1. **Theoretical Challenges**

   - Discrete Reduction

   - Geometric Integration

   - Quantum Systems

1. **Computational Challenges**

   - Structure Preservation

   - Long-Time Stability

   - Constraint Handling

## Advanced Theoretical Extensions

### Formal Symplectic Geometry Framework

**Definition** (Symplectic Manifold with Momentum Map): A tuple $(M, \omega, \Phi)$ where:

- $(M, \omega)$ is a symplectic manifold with symplectic form $\omega$

- $\Phi: M \to \mathfrak{g}^*$ is an equivariant momentum map for group action $G$

- The momentum map satisfies: $d\langle\Phi, \xi\rangle = i_{\xi_M}\omega$ for all $\xi \in \mathfrak{g}$

**Theorem** (Marsden-Weinstein Reduction): If $\mu \in \mathfrak{g}^*$ is a regular value of $\Phi$ and $G_\mu$ acts freely on $\Phi^{-1}(\mu)$, then:

$$M_\mu = \Phi^{-1}(\mu)/G_\mu$$

is a symplectic manifold with induced symplectic structure.

```python
import numpy as np

# One Verlet step for the pendulum H = p^2/2 - cos(q); check the map is symplectic:
# its Jacobian M satisfies M^T Omega M = Omega, i.e. det M = 1 in one degree of freedom.
def step(z, h=0.1):
    q, p = z
    p = p - 0.5 * h * np.sin(q)
    q = q + h * p
    p = p - 0.5 * h * np.sin(q)
    return np.array([q, p])

z0, eps = np.array([0.7, 0.3]), 1e-6          # illustrative point
M = np.column_stack([(step(z0 + eps * e) - step(z0 - eps * e)) / (2 * eps) for e in np.eye(2)])
Omega = np.array([[0.0, 1.0], [-1.0, 0.0]])
print("det M =", np.linalg.det(M), " symplectic error =", np.abs(M.T @ Omega @ M - Omega).max())
```

### Infinite-Dimensional Extensions

**Definition** (Infinite-Dimensional Hamiltonian System): A tuple $(M, \omega, H)$ where:

- $M$ is an infinite-dimensional manifold (typically a function space)

- $\omega$ is a weakly non-degenerate closed 2-form

- $H: M \to \mathbb{R}$ is a smooth Hamiltonian functional

## Related Topics

1. [[hamiltonian_mechanics|Hamiltonian Mechanics]]

1. [[lagrangian_mechanics|Lagrangian Mechanics]]

1. [[control_theory|Control Theory]]

1. [[quantum_mechanics|Quantum Mechanics]]

1. [[field_theory|Classical Field Theory]]

1. [[differential_geometry|Differential Geometry]]

1. [[lie_groups|Lie Groups and Lie Algebras]]

1. [[symplectic_geometry|Symplectic Geometry]]


## References

- Arnold, V. I. (1989). Mathematical Methods of Classical Mechanics (2nd ed.). Springer.
- Marsden, J. E., & Ratiu, T. S. (1999). Introduction to Mechanics and Symmetry (2nd ed.). Springer.
- Marsden, J., & Weinstein, A. (1974). Reduction of symplectic manifolds with symmetry. Reports on Mathematical Physics, 5(1), 121-130.
- Marsden, J. E., & West, M. (2001). Discrete mechanics and variational integrators. Acta Numerica, 10, 357-514.
