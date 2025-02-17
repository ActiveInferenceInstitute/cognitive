# Path Integrals in Cognitive Modeling

---
title: Path Integrals
type: mathematics
status: stable
created: 2024-02-06
tags:
  - mathematics
  - path_integrals
  - stochastic_processes
  - optimization
semantic_relations:
  - type: foundation_for
    links: 
      - [[knowledge_base/mathematics/expected_free_energy|Expected Free Energy]]
      - [[knowledge_base/cognitive/active_inference|Active Inference]]
  - type: implemented_by
    links:
      - [[docs/api/numerical_methods|Numerical Methods]]
      - [[docs/guides/implementation_patterns|Implementation Patterns]]
---

## Overview

Path integrals provide a mathematical framework for computing expectations over trajectories in state space, crucial for policy evaluation and planning in active inference. This document outlines key concepts and implementations.

## Core Mathematics

### Path Integral Definition
```math
Z = \int \mathcal{D}[x(t)] \exp(-S[x(t)])

where:
Z = Partition function
x(t) = Path/trajectory
S[x(t)] = Action functional
\mathcal{D}[x(t)] = Path measure
```

### Action Functional
```math
S[x(t)] = \int_{t_0}^{t_1} L(x(t), \dot{x}(t), t) dt

where:
L = Lagrangian
t_0, t_1 = Start and end times
```

## Implementation

### Path Sampling
```python
# @path_sampling
def sample_paths(
    initial_state: np.ndarray,
    dynamics: Callable,
    time_horizon: int,
    n_samples: int
) -> np.ndarray:
    """
    Sample paths using stochastic dynamics.
    
    Mathematics:
        - [[knowledge_base/mathematics/stochastic_processes|Stochastic Processes]]
        - [[knowledge_base/mathematics/importance_sampling|Importance Sampling]]
    Implementation:
        - [[docs/api/numerical_methods#path-sampling|Path Sampling]]
    """
    # Initialize paths
    paths = np.zeros((n_samples, time_horizon, initial_state.shape[0]))
    paths[:, 0] = initial_state
    
    # Generate trajectories
    for t in range(1, time_horizon):
        paths[:, t] = dynamics(paths[:, t-1])
    
    return paths
```

### Action Computation
```python
# @action_computation
def compute_action(
    path: np.ndarray,
    lagrangian: Callable,
    dt: float
) -> float:
    """
    Compute action functional along path.
    
    Mathematics:
        - [[knowledge_base/mathematics/action_principle|Action Principle]]
        - [[knowledge_base/mathematics/numerical_integration|Numerical Integration]]
    Implementation:
        - [[docs/api/numerical_methods#action-computation|Action Computation]]
    """
    # Compute velocities
    velocities = np.gradient(path, dt, axis=0)
    
    # Compute Lagrangian at each point
    L = np.array([lagrangian(x, v) for x, v in zip(path, velocities)])
    
    # Integrate action
    S = np.trapz(L, dx=dt)
    
    return S
```

## Applications

### Policy Evaluation
```python
# @policy_evaluation
def evaluate_policy_path(
    policy: Policy,
    model: GenerativeModel,
    time_horizon: int,
    n_samples: int
) -> float:
    """
    Evaluate policy using path integral.
    
    Mathematics:
        - [[knowledge_base/mathematics/expected_free_energy|Expected Free Energy]]
        - [[knowledge_base/mathematics/monte_carlo|Monte Carlo Methods]]
    Implementation:
        - [[docs/guides/implementation_patterns#policy-evaluation|Policy Evaluation]]
    """
    # Sample paths
    paths = sample_policy_paths(policy, model, time_horizon, n_samples)
    
    # Compute actions
    actions = np.array([compute_action(path, model.lagrangian) for path in paths])
    
    # Compute path integral estimate
    Z = np.mean(np.exp(-actions))
    
    return -np.log(Z)
```

### Trajectory Optimization
```python
# @trajectory_optimization
def optimize_trajectory(
    initial_state: np.ndarray,
    target_state: np.ndarray,
    dynamics: Callable,
    cost_function: Callable
) -> np.ndarray:
    """
    Optimize trajectory using path integral control.
    
    Mathematics:
        - [[knowledge_base/mathematics/optimal_control|Optimal Control]]
        - [[knowledge_base/mathematics/variational_principles|Variational Principles]]
    Implementation:
        - [[docs/api/optimization_methods#trajectory-optimization|Trajectory Optimization]]
    """
    # Initialize trajectory
    trajectory = initialize_trajectory(initial_state, target_state)
    
    # Optimize using path integral control
    while not converged:
        # Sample perturbations
        perturbed_paths = sample_perturbations(trajectory)
        
        # Compute costs
        costs = np.array([cost_function(path) for path in perturbed_paths])
        
        # Update trajectory
        trajectory = update_trajectory(perturbed_paths, costs)
    
    return trajectory
```

## Numerical Methods

### 1. Monte Carlo Integration
```python
# @monte_carlo
def monte_carlo_path_integral(
    integrand: Callable,
    measure: Callable,
    n_samples: int
) -> float:
    """
    Compute path integral using Monte Carlo.
    
    Mathematics:
        - [[knowledge_base/mathematics/monte_carlo|Monte Carlo Methods]]
        - [[knowledge_base/mathematics/importance_sampling|Importance Sampling]]
    """
    # Generate samples
    samples = measure.sample(n_samples)
    
    # Compute weights
    weights = np.array([integrand(s) for s in samples])
    
    # Estimate integral
    Z = np.mean(weights)
    
    return Z
```

### 2. Stochastic Optimization
```python
# @stochastic_optimization
def stochastic_path_optimization(
    objective: Callable,
    initial_path: np.ndarray,
    learning_rate: float
) -> np.ndarray:
    """
    Optimize path using stochastic methods.
    
    Mathematics:
        - [[knowledge_base/mathematics/stochastic_optimization|Stochastic Optimization]]
        - [[knowledge_base/mathematics/gradient_descent|Gradient Descent]]
    """
    current_path = initial_path.copy()
    
    while not converged:
        # Sample noise
        noise = sample_noise(current_path.shape)
        
        # Evaluate perturbed paths
        value_plus = objective(current_path + noise)
        value_minus = objective(current_path - noise)
        
        # Update path
        gradient = (value_plus - value_minus) / (2 * noise)
        current_path -= learning_rate * gradient
    
    return current_path
```

## Implementation Considerations

### 1. Numerical Stability
- Use log-space computations
- Handle boundary conditions
- Monitor convergence
- Validate trajectories

### 2. Computational Efficiency
- Parallel path sampling
- Adaptive step sizes
- Importance sampling
- Caching strategies

## Related Mathematics
- [[knowledge_base/mathematics/stochastic_processes|Stochastic Processes]]
- [[knowledge_base/mathematics/variational_calculus|Variational Calculus]]
- [[knowledge_base/mathematics/optimal_control|Optimal Control]]
- [[knowledge_base/mathematics/statistical_physics|Statistical Physics]]

## References
- [[knowledge_base/cognitive/active_inference|Active Inference]]
- [[knowledge_base/mathematics/expected_free_energy|Expected Free Energy]]
- [[docs/guides/implementation_patterns|Implementation Patterns]]
- [[docs/api/numerical_methods|Numerical Methods]] 