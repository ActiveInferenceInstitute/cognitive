---
title: Differential Equations
type: concept
status: stable
created: 2024-02-12
tags:
  - mathematics
  - computation
  - systems
semantic_relations:
  - type: foundation
    links: 
      - [[calculus]]
      - [[linear_algebra]]
  - type: relates
    links:
      - [[dynamical_systems]]
      - [[optimization_theory]]
      - [[control_theory]]
---

# Differential Equations

## Overview

Differential Equations describe relationships between functions and their derivatives. They are fundamental to modeling continuous-time systems in physics, engineering, and cognitive science.

## Core Concepts

### Ordinary Differential Equations
```math
\frac{dy}{dt} = f(t,y)
```
where:
- $y$ is state variable
- $t$ is time
- $f$ is vector field

### Partial Differential Equations
```math
\frac{\partial u}{\partial t} = F(t,x,u,\nabla u,\nabla^2 u)
```
where:
- $u$ is field variable
- $x$ is spatial coordinate
- $F$ is differential operator

## Implementation

### ODE Solver

```python
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Callable

class ODESolver:
    def __init__(self,
                 vector_field: Callable,
                 dt: float = 0.01):
        """Initialize ODE solver.
        
        Args:
            vector_field: Vector field function
            dt: Time step
        """
        self.vector_field = vector_field
        self.dt = dt
    
    def euler(self,
             y0: np.ndarray,
             t: np.ndarray) -> np.ndarray:
        """Euler integration.
        
        Args:
            y0: Initial state
            t: Time points
            
        Returns:
            y: State trajectory
        """
        y = np.zeros((len(t), *y0.shape))
        y[0] = y0
        
        for i in range(len(t)-1):
            dy = self.vector_field(t[i], y[i])
            y[i+1] = y[i] + self.dt * dy
        
        return y
    
    def rk4(self,
           y0: np.ndarray,
           t: np.ndarray) -> np.ndarray:
        """RK4 integration.
        
        Args:
            y0: Initial state
            t: Time points
            
        Returns:
            y: State trajectory
        """
        y = np.zeros((len(t), *y0.shape))
        y[0] = y0
        
        for i in range(len(t)-1):
            k1 = self.vector_field(t[i], y[i])
            k2 = self.vector_field(
                t[i] + self.dt/2,
                y[i] + self.dt/2 * k1
            )
            k3 = self.vector_field(
                t[i] + self.dt/2,
                y[i] + self.dt/2 * k2
            )
            k4 = self.vector_field(
                t[i] + self.dt,
                y[i] + self.dt * k3
            )
            
            y[i+1] = y[i] + self.dt/6 * (
                k1 + 2*k2 + 2*k3 + k4
            )
        
        return y
```

### Neural ODE

```python
class NeuralODE(nn.Module):
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int):
        """Initialize neural ODE.
        
        Args:
            state_dim: State dimension
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        # Vector field network
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self,
               t: torch.Tensor,
               y: torch.Tensor) -> torch.Tensor:
        """Compute vector field.
        
        Args:
            t: Time tensor
            y: State tensor
            
        Returns:
            dy: State derivative
        """
        # Concatenate time and state
        ty = torch.cat([
            t.expand(y.shape[0], 1),
            y
        ], dim=1)
        
        return self.net(ty)
    
    def integrate(self,
                 y0: torch.Tensor,
                 t: torch.Tensor,
                 method: str = 'rk4') -> torch.Tensor:
        """Integrate system.
        
        Args:
            y0: Initial state
            t: Time points
            method: Integration method
            
        Returns:
            y: State trajectory
        """
        dt = t[1] - t[0]
        y = [y0]
        
        for i in range(len(t)-1):
            if method == 'euler':
                # Euler step
                dy = self.forward(t[i], y[-1])
                y_next = y[-1] + dt * dy
            
            elif method == 'rk4':
                # RK4 step
                k1 = self.forward(t[i], y[-1])
                k2 = self.forward(
                    t[i] + dt/2,
                    y[-1] + dt/2 * k1
                )
                k3 = self.forward(
                    t[i] + dt/2,
                    y[-1] + dt/2 * k2
                )
                k4 = self.forward(
                    t[i] + dt,
                    y[-1] + dt * k3
                )
                
                y_next = y[-1] + dt/6 * (
                    k1 + 2*k2 + 2*k3 + k4
                )
            
            y.append(y_next)
        
        return torch.stack(y)
```

### Training Loop

```python
def train_node(node: NeuralODE,
              dataset: torch.Tensor,
              n_epochs: int = 100,
              learning_rate: float = 0.01) -> List[float]:
    """Train neural ODE.
    
    Args:
        node: Neural ODE
        dataset: Training data
        n_epochs: Number of epochs
        learning_rate: Learning rate
        
    Returns:
        losses: Training losses
    """
    optimizer = torch.optim.Adam(
        node.parameters(),
        lr=learning_rate
    )
    losses = []
    
    for epoch in range(n_epochs):
        total_loss = 0
        
        for trajectory in dataset:
            # Get time points
            t = torch.linspace(
                0, 1, len(trajectory)
            )
            
            # Forward pass
            y_pred = node.integrate(
                trajectory[0], t
            )
            
            # Compute loss
            loss = torch.mean(
                (trajectory - y_pred)**2
            )
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataset)
        losses.append(avg_loss)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    return losses
```

## Best Practices

### Model Design
1. Choose appropriate method
2. Design vector field
3. Initialize parameters
4. Consider stability

### Implementation
1. Monitor error
2. Handle stiffness
3. Validate solutions
4. Test convergence

### Training
1. Tune learning rates
2. Balance accuracy
3. Monitor stability
4. Validate solutions

## Common Issues

### Technical Challenges
1. Numerical instability
2. Stiffness problems
3. Error accumulation
4. Convergence issues

### Solutions
1. Adaptive stepping
2. Implicit methods
3. Error control
4. Method selection

## Related Documentation
- [[calculus]]
- [[linear_algebra]]
- [[dynamical_systems]] 