---
title: Calculus
type: mathematical_concept
status: stable
created: 2024-02-12
tags:
- mathematics
- computation
- foundations
semantic_relations:
  - type: foundation
    links:
      - "[[real_analysis]]"
      - "[[linear_algebra]]"
  - type: relates
    links:
      - "[[differential_equations]]"
      - "[[optimization_theory]]"
      - "[[numerical_methods]]"
---

# Calculus

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

## Overview

Calculus is the mathematical study of continuous change. It provides the foundation for understanding rates of change, accumulation, and optimization in physical and cognitive systems.

## Core Concepts

### Differentiation

```math

\frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}

```

where:

- $f$ is function

- $x$ is variable

- $h$ is step size

### Integration

```math

\int_a^b f(x)dx = \lim_{n \to \infty} \sum_{i=1}^n f(x_i)\Delta x

```

where:

- $[a,b]$ is interval

- $\Delta x$ is partition size

- $x_i$ are sample points

## Implementation

### Numerical Differentiation

```python
import numpy as np

def forward_difference(f, x, h=1e-6):
    return (f(x + h) - f(x)) / h  # truncation error O(h)

def central_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)  # truncation error O(h^2)

x = np.linspace(0.0, 2.0, 5)  # illustrative grid
print(np.max(np.abs(forward_difference(np.sin, x) - np.cos(x))))
print(np.max(np.abs(central_difference(np.sin, x) - np.cos(x))))
```

### Numerical Integration

```python
import numpy as np

def trapezoid(f, a, b, n=100):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (b - a) / n * (y.sum() - 0.5 * (y[0] + y[-1]))

def simpson(f, a, b, n=100):  # n must be even
    x = np.linspace(a, b, n + 1)
    y = f(x)
    odd, even = y[1:-1:2].sum(), y[2:-1:2].sum()
    return (b - a) / (3 * n) * (y[0] + y[-1] + 4 * odd + 2 * even)

# The integral of sin over [0, pi] is exactly 2.
print(trapezoid(np.sin, 0, np.pi) - 2, simpson(np.sin, 0, np.pi) - 2)
```

### Automatic Differentiation

```python
import math

class Dual:
    """Forward-mode automatic differentiation: carries (value, derivative)."""

    def __init__(self, val, der=0.0):
        self.val, self.der = val, der

    @staticmethod
    def lift(o):
        return o if isinstance(o, Dual) else Dual(o)

    def __add__(self, o):
        o = Dual.lift(o)
        return Dual(self.val + o.val, self.der + o.der)

    def __mul__(self, o):
        o = Dual.lift(o)
        return Dual(self.val * o.val, self.der * o.val + self.val * o.der)  # product rule

    __radd__, __rmul__ = __add__, __mul__

def sin(d):
    return Dual(math.sin(d.val), math.cos(d.val) * d.der)  # chain rule

f = lambda x: x * sin(x) + 3 * x
out = f(Dual(1.5, 1.0))  # seed dx/dx = 1
print(out.der, math.sin(1.5) + 1.5 * math.cos(1.5) + 3)  # AD vs hand-derived f'(1.5)
```

## Best Practices

### Method Selection

1. Choose appropriate method

1. Consider accuracy needs

1. Handle edge cases

1. Monitor stability

### Implementation

1. Use stable algorithms

1. Handle singularities

1. Validate results

1. Test convergence

### Optimization

1. Balance accuracy

1. Consider efficiency

1. Monitor errors

1. Validate solutions

## Common Issues

### Technical Challenges

1. Numerical instability

1. Roundoff errors

1. Singularities

1. Convergence issues

### Solutions

1. Adaptive methods

1. Error control

1. Regularization

1. Method selection

## Advanced Topics

### Vector Calculus

```math

\nabla f = (\frac{\partial f}{\partial x_1}, ..., \frac{\partial f}{\partial x_n})

```

where:

- $\nabla f$ is gradient

- $x_i$ are coordinates

- $f$ is scalar field

### Differential Forms

```math

\omega = \sum_{i_1<...<i_k} a_{i_1...i_k}dx_{i_1}\wedge...\wedge dx_{i_k}

```

where:

- $\omega$ is k-form

- $a_{i_1...i_k}$ are coefficients

- $\wedge$ is wedge product

## Applications

### Optimization

- [[optimization_theory|Optimization Theory]]

  - Gradient descent

  - Newton's method

  - Natural gradients

### Differential Equations

- [[differential_equations|Differential Equations]]

  - ODEs

  - PDEs

  - Numerical methods

### Information Theory

- [[information_theory|Information Theory]]

  - Entropy gradients

  - Fisher information

  - KL divergence

### Machine Learning

- [[machine_learning|Machine Learning]]

  - Backpropagation

  - Neural ODEs

  - Gradient flows

## Theoretical Connections

### Analysis

- [[real_analysis|Real Analysis]]

  - Limits and continuity

  - Measure theory

  - Function spaces

### Geometry

- [[differential_geometry|Differential Geometry]]

  - Manifolds

  - Tangent spaces

  - Curvature

### Probability

- [[probability_theory|Probability Theory]]

  - Stochastic calculus

  - Ito integration

  - Fokker-Planck equations

## Computational Methods

## Related Documentation

- [[real_analysis|Real Analysis]]

- [[linear_algebra|Linear Algebra]]

- [[numerical_methods|Numerical Methods]]

- [[optimization_theory|Optimization Theory]]

- [[differential_equations|Differential Equations]]

- [[probability_theory|Probability Theory]]

- [[information_theory|Information Theory]]

- [[machine_learning|Machine Learning]]

## Multivariate Calculus

### Partial Derivatives

```math

\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x + he_i) - f(x)}{h}

```

where:

- $e_i$ is unit vector

- $x$ is position vector

- $h$ is step size

### Chain Rule

```math

\frac{\partial f}{\partial x} = \sum_i \frac{\partial f}{\partial u_i} \frac{\partial u_i}{\partial x}

```

where:

- $u_i$ are intermediate variables

- $f$ is composite function

### Implementation

```python
import numpy as np

def jacobian(F, x, h=1e-6):
    x = np.asarray(x, float)
    cols = []
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = h
        cols.append((F(x + e) - F(x - e)) / (2 * h))
    return np.stack(cols, axis=1)  # J[i, j] = dF_i / dx_j

F = lambda x: np.array([x[0] * x[1], np.sin(x[0]) + x[1] ** 2])
x0 = np.array([0.7, -1.2])  # illustrative point
J_exact = np.array([[x0[1], x0[0]], [np.cos(x0[0]), 2 * x0[1]]])
print(np.allclose(jacobian(F, x0), J_exact, atol=1e-6))

# Chain rule: the Jacobian of g(F(x)) is J_g(F(x)) @ J_F(x).
G = lambda x: np.array([F(x)[0] + F(x)[1] ** 2])
Jg = np.array([[1.0, 2 * F(x0)[1]]])
print(np.allclose(jacobian(G, x0), Jg @ J_exact, atol=1e-5))
```

## Tensor Calculus

## Applications in Physics

### Hamiltonian Mechanics

- [[dynamical_systems|Dynamical Systems]]

  - Phase space

  - Canonical transformations

  - Hamilton's equations

### Field Theory

- [[quantum_field_theory|Quantum Field Theory]]

  - Functional derivatives

  - Path integrals

  - Gauge theory

### General Relativity

- [[differential_geometry|Differential Geometry]]

  - Metric tensors

  - Geodesics

  - Einstein equations

## Computational Applications

### Scientific Computing

- [[numerical_methods|Numerical Methods]]

  - Finite differences

  - Spectral methods

  - Mesh-free methods

### Machine Learning

- [[deep_learning|Deep Learning]]

  - Backpropagation

  - Natural gradients

  - Information geometry

### Control Theory

- [[control_theory|Control Theory]]

  - Optimal control

  - Lyapunov theory

  - Feedback systems

## Related Documentation

- [[real_analysis|Real Analysis]]

- [[differential_geometry|Differential Geometry]]

- [[dynamical_systems|Dynamical Systems]]

- [[quantum_field_theory|Quantum Field Theory]]

- [[control_theory|Control Theory]]

- [[deep_learning|Deep Learning]]

## Complex Analysis

### Complex Differentiation

```math

f'(z) = \lim_{h \to 0} \frac{f(z + h) - f(z)}{h}

```

where:

- $z$ is complex variable

- $f$ is complex function

- $h$ approaches 0 in complex plane

### Cauchy-Riemann Equations

```math

\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}, \quad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}

```

where:

- $f(z) = u(x,y) + iv(x,y)$

- $u$ is real part

- $v$ is imaginary part

## Functional Analysis

### Applications

## Applications in Signal Processing

### Wavelet Analysis

- [[signal_processing|Signal Processing]]

  - Wavelet transforms

  - Multiresolution analysis

  - Filter banks

### Harmonic Analysis

- Harmonic Analysis

  - Fourier transforms

  - Spectral theory

  - Group representations

### Time-Frequency Analysis

- [[time_frequency_analysis|Time-Frequency Analysis]]

  - Short-time Fourier transform

  - Gabor transform

  - Wigner distribution

## Applications in Quantum Mechanics

### Hilbert Spaces

- [[functional_analysis|Functional Analysis]]

  - State vectors

  - Observables

  - Unitary evolution

### Operator Theory

- Operator Theory

  - Self-adjoint operators

  - Spectral theory

  - Von Neumann algebras

## Related Documentation

- Complex Analysis

- [[functional_analysis|Functional Analysis]]

- Harmonic Analysis

- Operator Theory

- [[signal_processing|Signal Processing]]

- [[quantum_mechanics|Quantum Mechanics]]

## Variational Calculus

### Euler-Lagrange Equations

```math

\frac{d}{dt}\frac{\partial L}{\partial \dot{q}} - \frac{\partial L}{\partial q} = 0

```

where:

- $L$ is Lagrangian

- $q$ is generalized coordinate

- $\dot{q}$ is time derivative

## Geometric Calculus

## Applications in Machine Learning

### Neural Differential Equations

- [[neural_odes|Neural ODEs]]

  - Continuous-depth networks

  - Adjoint sensitivity

  - Normalizing flows

### Geometric Deep Learning

- Geometric Deep Learning

  - Manifold learning

  - Graph neural networks

  - Gauge equivariance

### Information Geometry

- [[information_geometry|Information Geometry]]

  - Statistical manifolds

  - Natural gradients

  - Fisher metrics

## Related Documentation

- [[variational_methods|Variational Methods]]

- [[geometric_algebra|Geometric Algebra]]

- [[neural_odes|Neural ODEs]]

- Geometric Deep Learning

- [[information_geometry|Information Geometry]]

## Symplectic Geometry

### Symplectic Form

```math

\omega = \sum_{i=1}^n dp_i \wedge dq_i

```

where:

- $p_i$ are momentum coordinates

- $q_i$ are position coordinates

- $\wedge$ is wedge product

## Stochastic Calculus

### Ito Process

```math

dX_t = \mu(X_t,t)dt + \sigma(X_t,t)dW_t

```

where:

- $X_t$ is stochastic process

- $\mu$ is drift term

- $\sigma$ is diffusion term

- $W_t$ is Wiener process

## Applications in Machine Learning

### Stochastic Optimization

- [[stochastic_optimization|Stochastic Optimization]]

  - SGD with momentum

  - Natural gradient descent

  - Stochastic Hamilton Monte Carlo

### Neural SDEs

- Neural SDEs

  - Continuous-time latent models

  - Stochastic adjoint sensitivity

  - Stochastic normalizing flows

### Statistical Learning

- [[statistical_learning|Statistical Learning]]

  - Stochastic approximation

  - Online learning

  - Diffusion models

## Related Documentation

- [[symplectic_geometry|Symplectic Geometry]]

- [[stochastic_processes|Stochastic Processes]]

- [[financial_mathematics|Financial Mathematics]]

- Neural SDEs

- [[stochastic_optimization|Stochastic Optimization]]

## Applications in Cognitive Science

### Predictive Processing

```python
import numpy as np

# Gaussian model with illustrative values: y ~ N(g(mu), 1/pi_y), mu ~ N(m, 1/pi_p).
g = np.tanh
y, m, pi_y, pi_p = 0.8, 0.0, 4.0, 1.0

def free_energy(mu):
    """Precision-weighted squared prediction errors (Laplace form, constants dropped)."""
    return 0.5 * pi_y * (y - g(mu)) ** 2 + 0.5 * pi_p * (mu - m) ** 2

def grad_free_energy(mu):
    dg = 1 - np.tanh(mu) ** 2  # g'(mu)
    return -pi_y * (y - g(mu)) * dg + pi_p * (mu - m)

h = 1e-6
check = (free_energy(0.3 + h) - free_energy(0.3 - h)) / (2 * h)
print(np.isclose(grad_free_energy(0.3), check))  # analytic gradient vs central difference

mu = 0.0
for _ in range(500):  # gradient descent on F: belief updating
    mu -= 0.05 * grad_free_energy(mu)
print(mu, grad_free_energy(mu))  # gradient is ~0 at the minimum
```

## Related Documentation

- [[predictive_coding|Predictive Coding]]

- [[active_inference|Active Inference]]

- [[belief_updating|Belief Updating]]

- [[precision_weighting|Precision Weighting]]

- [[free_energy_principle|Free Energy Principle]]

## References
- Rudin, W. (1976). Principles of Mathematical Analysis (3rd ed.). McGraw-Hill.
- Arnold, V. I. (1989). Mathematical Methods of Classical Mechanics (2nd ed.). Springer.
- Oksendal, B. (2003). Stochastic Differential Equations: An Introduction with Applications (6th ed.). Springer.
- Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning: A survey. Journal of Machine Learning Research, 18(153), 1-43.
- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637-654.
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior. MIT Press.
