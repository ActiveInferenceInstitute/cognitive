---
title: Control Theory
type: concept
status: stable
created: 2024-03-15
complexity: advanced
processing_priority: 1
tags:
  - mathematics
  - systems
  - dynamics
  - optimization
  - feedback
semantic_relations:
  - type: foundation_for
    links:
      - [[active_inference]]
      - [[optimal_control]]
      - [[adaptive_control]]
  - type: implements
    links:
      - [[dynamical_systems]]
      - [[optimization_theory]]
      - [[differential_equations]]
  - type: relates
    links:
      - [[statistical_physics]]
      - [[information_theory]]
      - [[free_energy_principle]]
      - [[stochastic_processes]]

---

# Control Theory

## Overview

Control Theory provides a mathematical framework for analyzing and designing systems that regulate their behavior through feedback mechanisms. It forms a crucial bridge between dynamical systems theory and cognitive science, particularly in understanding how biological systems maintain homeostasis and how agents perform active inference.

## Mathematical Foundation

### State-Space Representation

#### Linear Systems
```math
\begin{align*}
\dot{x} &= Ax + Bu \\
y &= Cx + Du
\end{align*}
```
where:
- $x$ is state vector
- $u$ is control input
- $y$ is output
- $A,B,C,D$ are system matrices

#### Nonlinear Systems
```math
\begin{align*}
\dot{x} &= f(x,u,t) \\
y &= h(x,u,t)
\end{align*}
```

### Optimal Control

#### Cost Function
```math
J = \int_0^T L(x,u,t)dt + \phi(x(T))
```
where:
- $L$ is running cost
- $\phi$ is terminal cost
- $T$ is time horizon

#### Hamilton-Jacobi-Bellman Equation
```math
-\frac{\partial V}{\partial t} = \min_u \left\{L(x,u,t) + \nabla V \cdot f(x,u,t)\right\}
```
where:
- $V$ is value function
- $\nabla V$ is value gradient

## Implementation

### Linear Control System

```python
class LinearControlSystem:
    def __init__(self,
                 A: np.ndarray,
                 B: np.ndarray,
                 C: np.ndarray,
                 D: Optional[np.ndarray] = None):
        """Initialize linear control system.
        
        Args:
            A: State transition matrix
            B: Input matrix
            C: Output matrix
            D: Feedthrough matrix
        """
        self.A = A
        self.B = B
        self.C = C
        self.D = D if D is not None else np.zeros((C.shape[0], B.shape[1]))
        
        # System dimensions
        self.n_states = A.shape[0]
        self.n_inputs = B.shape[1]
        self.n_outputs = C.shape[0]
        
        # Initialize state
        self.state = np.zeros(self.n_states)
    
    def step(self,
            u: np.ndarray,
            dt: float = 0.01) -> np.ndarray:
        """Simulate one step of system dynamics.
        
        Args:
            u: Control input
            dt: Time step
            
        Returns:
            y: System output
        """
        # Update state
        self.state += dt * (self.A @ self.state + self.B @ u)
        
        # Compute output
        y = self.C @ self.state + self.D @ u
        
        return y
    
    def simulate(self,
                u_trajectory: np.ndarray,
                dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate system trajectory.
        
        Args:
            u_trajectory: Control input trajectory
            dt: Time step
            
        Returns:
            x_trajectory: State trajectory
            y_trajectory: Output trajectory
        """
        n_steps = len(u_trajectory)
        x_trajectory = np.zeros((n_steps, self.n_states))
        y_trajectory = np.zeros((n_steps, self.n_outputs))
        
        for t in range(n_steps):
            y_trajectory[t] = self.step(u_trajectory[t], dt)
            x_trajectory[t] = self.state
        
        return x_trajectory, y_trajectory
```

### Optimal Controller

```python
class OptimalController:
    def __init__(self,
                 system: LinearControlSystem,
                 Q: np.ndarray,
                 R: np.ndarray):
        """Initialize optimal controller.
        
        Args:
            system: Linear control system
            Q: State cost matrix
            R: Control cost matrix
        """
        self.system = system
        self.Q = Q
        self.R = R
        
        # Solve Riccati equation
        self.P = self.solve_riccati()
        
        # Compute optimal gain
        self.K = self.compute_optimal_gain()
    
    def solve_riccati(self) -> np.ndarray:
        """Solve algebraic Riccati equation."""
        A, B = self.system.A, self.system.B
        
        def riccati_rhs(P):
            return (A.T @ P + P @ A - 
                   P @ B @ np.linalg.inv(self.R) @ B.T @ P + 
                   self.Q)
        
        # Initialize
        P = self.Q
        
        # Iterate until convergence
        for _ in range(1000):
            P_new = P - 0.01 * riccati_rhs(P)
            if np.allclose(P, P_new):
                break
            P = P_new
        
        return P
    
    def compute_optimal_gain(self) -> np.ndarray:
        """Compute optimal feedback gain."""
        return np.linalg.inv(self.R) @ self.system.B.T @ self.P
    
    def compute_control(self,
                       x: np.ndarray) -> np.ndarray:
        """Compute optimal control input.
        
        Args:
            x: Current state
            
        Returns:
            u: Optimal control input
        """
        return -self.K @ x
```

### Adaptive Controller

```python
class AdaptiveController:
    def __init__(self,
                 system_dim: int,
                 learning_rate: float = 0.1):
        """Initialize adaptive controller.
        
        Args:
            system_dim: System dimension
            learning_rate: Parameter adaptation rate
        """
        self.dim = system_dim
        self.lr = learning_rate
        
        # Initialize parameters
        self.params = np.zeros((system_dim, system_dim))
        self.covariance = np.eye(system_dim)
    
    def update(self,
              x: np.ndarray,
              y: np.ndarray,
              y_desired: np.ndarray) -> np.ndarray:
        """Update controller parameters.
        
        Args:
            x: Current state
            y: Current output
            y_desired: Desired output
            
        Returns:
            u: Control input
        """
        # Compute error
        error = y_desired - y
        
        # Update covariance
        self.covariance = self.update_covariance(x)
        
        # Update parameters
        self.params += self.lr * np.outer(
            self.covariance @ error,
            x
        )
        
        # Compute control
        u = self.params @ x
        
        return u
    
    def update_covariance(self,
                         x: np.ndarray) -> np.ndarray:
        """Update parameter covariance.
        
        Args:
            x: Current state
            
        Returns:
            P: Updated covariance
        """
        P = self.covariance
        
        # RLS update
        k = P @ x / (1 + x.T @ P @ x)
        P = P - np.outer(k, x.T @ P)
        
        return P
```

## Applications

### Physical Systems

#### Mechanical Systems
- Robot control
- Vehicle dynamics
- Vibration damping
- Trajectory tracking

#### Process Control
- Chemical reactions
- Temperature regulation
- Flow control
- Pressure systems

### Biological Systems

#### Neural Control
- Motor control
- Sensory processing
- Homeostatic regulation
- Learning dynamics

#### Cognitive Control
- Active inference
- Decision making
- Attention allocation
- Behavioral adaptation

### Information Processing

#### Filtering
- Kalman filtering
- Particle filtering
- State estimation
- Uncertainty propagation

#### Learning Control
- Reinforcement learning
- Adaptive control
- Neural control
- Model predictive control

## Advanced Topics

### Robust Control
- H∞ control
- μ-synthesis
- Lyapunov stability
- Uncertainty modeling

### Stochastic Control
- Linear quadratic Gaussian
- Partially observable MDPs
- Risk-sensitive control
- Path integral control

### Nonlinear Control
- Feedback linearization
- Backstepping
- Sliding mode control
- Adaptive control

## Best Practices

### Design
1. System identification
2. Controller selection
3. Parameter tuning
4. Stability analysis

### Implementation
1. Discretization
2. Numerical stability
3. Anti-windup
4. Noise handling

### Validation
1. Stability margins
2. Performance metrics
3. Robustness tests
4. Uncertainty analysis

## Common Issues

### Technical Challenges
1. Model uncertainty
2. Measurement noise
3. Time delays
4. Nonlinearities

### Solutions
1. Robust design
2. Adaptive methods
3. Filtering techniques
4. Gain scheduling

## Related Documentation
- [[dynamical_systems]]
- [[optimization_theory]]
- [[statistical_physics]]
- [[active_inference]]
- [[optimal_control]]
- [[adaptive_control]] 