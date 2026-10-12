---
title: Continuous Time Modeling in Active Inference and Predictive Processing
type: concept
status: stable
---

# Continuous Time Modeling in Active Inference and Predictive Processing

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

## Introduction

Continuous time modeling in Active Inference and predictive processing provides a mathematical framework for understanding how biological systems continuously interact with their environment through perception and action. This document outlines the key concepts, mathematical foundations, and implementations of continuous-time active inference.

## Mathematical Foundations

### Continuous-Time Free Energy Principle

The Free Energy Principle (FEP) in continuous time is formulated as a path integral of variational free energy over time:

```math

F = ∫ F(x(t), μ(t), a(t)) dt

```

where:

- x(t) represents the true hidden states

- μ(t) represents internal states (expectations)

- a(t) represents actions

The variational free energy can be decomposed into:

```math

F = E_q[ln q(x) - ln p(x,y)] = D_KL[q(x)||p(x|y)] - ln p(y)

```

where:

- q(x) is the recognition density

- p(x,y) is the generative model

- D_KL is the Kullback-Leibler divergence

### Generative Model

The continuous-time generative model is defined by stochastic differential equations:

```math

dx = f(x,v,θ)dt + σ_x dW_x

dy = g(x,θ)dt + σ_y dW_y

```

where:

- f(x,v,θ) is the flow function describing dynamics

- g(x,θ) is the observation function

- σ_x, σ_y are noise terms

- dW_x, dW_y are Wiener processes

#### Extended Generative Model Components

1. **Flow Function Decomposition**:

```math

f(x,v,θ) = f_0(x,θ) + f_1(x,θ)v + f_2(x,θ)v^2 + ...

```

1. **Hierarchical Structure**:

```math

dx_i = f_i(x_i, x_{i+1})dt + σ_i dW_i

```

1. **Precision Parameters**:

```math

Π_x = (σ_x σ_x^T)^{-1}

Π_y = (σ_y σ_y^T)^{-1}

```

## Core Components

### 1. State Estimation

Continuous-time state estimation involves solving:

```math

dμ/dt = Dμ - ∂F/∂μ

```

where:

- Dμ is a temporal derivative operator

- ∂F/∂μ is the gradient of free energy

#### Recognition Dynamics

The recognition dynamics can be expanded as:

```math

dμ/dt = Dμ - (∂_μ ε_x^T Π_x ε_x + ∂_μ ε_y^T Π_y ε_y)

```

where:

- ε_x = Dμ - f(μ,v,θ) (dynamics prediction error)

- ε_y = y - g(μ,θ) (sensory prediction error)

### 2. Action Selection

Action selection in continuous time follows:

```math

da/dt = -∂F/∂a = -∂_a ε_y^T Π_y ε_y

```

#### Active Inference Implementation

1. **Sensorimotor Integration**:

```python
import numpy as np

def action_gradient(y, mu, g, Pi_y, dy_da):
    """-dF/da = -(dy/da)' Pi_y eps_y, with eps_y = y - g(mu)."""
    eps_y = y - g(mu)
    return -dy_da.T @ Pi_y @ eps_y
```

1. **Action Optimization**:

```python
def update_action(a, grad, dt, a_max=5.0):
    return np.clip(a + dt * grad, -a_max, a_max)
```

## Implementation Considerations

### 1. Numerical Integration

#### Runge-Kutta 4th Order Implementation

```python
def rk4_step(state, dt, dynamics):
    k1 = dynamics(state)
    k2 = dynamics(state + dt * k1 / 2)
    k3 = dynamics(state + dt * k2 / 2)
    k4 = dynamics(state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

state = np.array([1.0])
for _ in range(10):                       # dx/dt = -x over t in [0, 1]
    state = rk4_step(state, 0.1, lambda x: -x)
print(state, np.exp(-1.0))                # RK4 vs exact solution
```

#### Adaptive Step Size Control

```python
def adaptive_step(state, dt, dynamics, tol=1e-6):
    """Halve dt until one full step and two half steps agree to within tol."""
    while True:
        full = rk4_step(state, dt, dynamics)
        half = rk4_step(rk4_step(state, dt / 2, dynamics), dt / 2, dynamics)
        if np.max(np.abs(full - half)) < tol:
            return half, dt
        dt /= 2

print(adaptive_step(np.array([1.0]), 1.0, lambda x: -50 * x))
```

### 2. Precision Engineering

#### Dynamic Precision Updates

```python
def update_precision(Pi, error, learning_rate=0.1, ridge=1e-3):
    """Track the error covariance with a running average and set Pi to its inverse."""
    Sigma = np.linalg.inv(Pi) + learning_rate * (np.outer(error, error) - np.linalg.inv(Pi))
    return np.linalg.inv(Sigma + ridge * np.eye(len(error)))

print(update_precision(np.eye(2), np.array([0.5, -0.2])))
```

## Practical Implementation

### Complete Agent Implementation

```python
class ContinuousTimeAgent:
    """Gradient descent on F = 1/2 eps_y' Pi_y eps_y + 1/2 eps_x' Pi_x eps_x in
    generalised coordinates (mu, mu'), with linear g(mu) = C mu and f(mu) = A mu + b."""

    def __init__(self, A, b, C, Pi_x, Pi_y):
        self.A, self.b, self.C, self.Pi_x, self.Pi_y = A, b, C, Pi_x, Pi_y
        n = A.shape[0]
        self.mu, self.mu1, self.action = np.zeros(n), np.zeros(n), np.zeros(n)

    def errors(self, y):
        return y - self.C @ self.mu, self.mu1 - (self.A @ self.mu + self.b)

    def free_energy(self, y):
        e_y, e_x = self.errors(y)
        return 0.5 * (e_y @ self.Pi_y @ e_y + e_x @ self.Pi_x @ e_x)

    def step(self, y, dt):
        e_y, e_x = self.errors(y)
        dF_dmu = -self.C.T @ self.Pi_y @ e_y - self.A.T @ self.Pi_x @ e_x
        dF_dmu1 = self.Pi_x @ e_x
        self.mu = self.mu + dt * (self.mu1 - dF_dmu)          # d mu/dt = D mu - dF/dmu
        self.mu1 = self.mu1 - dt * dF_dmu1
        self.action = update_action(self.action, action_gradient(
            y, self.mu, lambda m: self.C @ m, self.Pi_y, np.eye(len(y))), dt)
        return self.action

# Illustrative world: dx/dt = -x + a, y = x. The agent expects x to settle at 1.
n = 1
agent = ContinuousTimeAgent(A=-np.eye(n), b=np.ones(n), C=np.eye(n),
                            Pi_x=np.eye(n), Pi_y=np.eye(n))
x, dt = np.zeros(n), 0.01
for _ in range(2000):
    a = agent.step(x, dt)
    x = x + dt * (-x + a)
print("hidden state:", x, "belief:", agent.mu)
```

## Advanced Topics

### 1. Information Geometry

The statistical manifold of the generative model can be characterized by the Fisher information metric:

```math

g_{ij}(θ) = E_p[-∂^2 ln p(x,y|θ)/∂θ_i ∂θ_j]

```

### 2. Stochastic Integration

For systems with significant noise, the Fokker-Planck equation describes the evolution of the probability density:

```math

∂p/∂t = -∇·(fp) + (1/2)∇·(D∇p)

```

where D is the diffusion tensor.

## Testing Framework

```python
agent = ContinuousTimeAgent(A=-np.eye(2), b=np.ones(2), C=np.eye(2),
                            Pi_x=np.eye(2), Pi_y=np.eye(2))
y = np.random.default_rng(0).standard_normal(2)
F0 = agent.free_energy(y)
for _ in range(100):
    agent.step(y, dt=0.01)
assert agent.free_energy(y) < F0, "free energy should fall under gradient descent"
```

## References

1. Friston, K. J., et al. (2010). Action and behavior: a free-energy formulation

1. Buckley, C. L., et al. (2017). The free energy principle for action and perception

1. Da Costa, L., et al. (2020). Active inference on continuous time: a real-time implementation

1. Baltieri, M., & Buckley, C. L. (2019). PID control as a process of active inference

1. Isomura, T., & Friston, K. (2018). In vitro neural networks minimize variational free energy

1. Tschantz, A., et al. (2020). Learning action-oriented models through active inference

1. Millidge, B., et al. (2021). Neural active inference: Deep learning of prediction, action, and precision

1. Bogacz, R. (2017). A tutorial on the free-energy framework for modelling perception and learning

## Future Directions

1. **Theoretical Extensions**

   - Non-Gaussian generative models

   - Mixed discrete-continuous systems

   - Stochastic control theory integration

   - Information geometry applications

   - Quantum active inference

1. **Implementation Advances**

   - Real-time implementations

   - Distributed computing approaches

   - Neural network approximations

   - Quantum computing implementations

   - Neuromorphic hardware optimization

1. **Applications**

   - Robotics control

   - Neural modeling

   - Adaptive systems

   - Brain-computer interfaces

   - Autonomous vehicles

   - Climate modeling

   - Financial systems

   - Social systems modeling

## Advanced Implementation Frameworks

### 1. Path Integral Control Integration

The path integral formulation of active inference combines continuous-time dynamics with optimal control:

```math

P(τ|π) ∝ exp(-S[τ]/λ)

```

where:

- τ is the trajectory

- S[τ] is the path cost

- λ is the temperature parameter

## Advanced Optimization Techniques

### 1. Natural Gradient Descent

The natural gradient provides a more efficient optimization by accounting for the information geometry:

```python
# Natural gradient: precondition by the inverse metric G (here G = H, illustrative)
H = np.diag([100.0, 1.0])                   # badly scaled quadratic F = 1/2 theta' H theta
theta = np.array([1.0, 1.0])
plain = theta - 0.01 * (H @ theta)
natural = theta - np.linalg.solve(H, H @ theta)
print(plain, natural)                       # natural step reaches the minimum at 0
```

