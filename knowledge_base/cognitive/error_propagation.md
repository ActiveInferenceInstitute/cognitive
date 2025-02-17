---
title: Error Propagation
type: concept
status: stable
created: 2024-02-12
tags:
  - cognitive
  - neuroscience
  - computation
semantic_relations:
  - type: foundation
    links: 
      - [[predictive_coding]]
      - [[hierarchical_inference]]
  - type: relates
    links:
      - [[free_energy_principle]]
      - [[precision_weighting]]
      - [[active_inference]]
---

# Error Propagation

## Overview

Error Propagation is a fundamental mechanism in predictive processing networks that describes how prediction errors flow through hierarchical layers. It involves both bottom-up propagation of prediction errors and top-down propagation of predictions.

## Core Concepts

### Bottom-up Error Flow
```math
ε_l^↑ = x_l - g_l(x_{l+1})
```
where:
- $ε_l^↑$ is bottom-up error at level $l$
- $x_l$ is state at level $l$
- $g_l$ is generative function

### Top-down Prediction Flow
```math
ε_l^↓ = x_l - f_l(x_{l-1})
```
where:
- $ε_l^↓$ is top-down error at level $l$
- $f_l$ is recognition function

## Implementation

### Error Propagation Layer

```python
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class ErrorPropagationLayer(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int):
        """Initialize error propagation layer.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden dimension
            output_size: Output dimension
        """
        super().__init__()
        
        # Forward model (bottom-up)
        self.forward_model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Backward model (top-down)
        self.backward_model = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        
        # Error scaling
        self.forward_scale = nn.Parameter(
            torch.ones(output_size)
        )
        self.backward_scale = nn.Parameter(
            torch.ones(input_size)
        )
    
    def forward(self,
               bottom_input: torch.Tensor,
               top_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through layer.
        
        Args:
            bottom_input: Input from lower level
            top_input: Input from higher level
            
        Returns:
            forward_error: Forward prediction error
            backward_error: Backward prediction error
        """
        # Forward prediction
        forward_pred = self.forward_model(bottom_input)
        
        # Backward prediction
        if top_input is not None:
            backward_pred = self.backward_model(top_input)
        else:
            backward_pred = self.backward_model(forward_pred)
        
        # Compute errors
        forward_error = (
            (forward_pred - (top_input if top_input is not None else forward_pred)) *
            self.forward_scale
        )
        backward_error = (
            (bottom_input - backward_pred) *
            self.backward_scale
        )
        
        return forward_error, backward_error
```

### Error Propagation Network

```python
class ErrorPropagationNetwork(nn.Module):
    def __init__(self,
                 layer_sizes: List[int],
                 hidden_sizes: List[int]):
        """Initialize error propagation network.
        
        Args:
            layer_sizes: List of layer sizes
            hidden_sizes: List of hidden sizes
        """
        super().__init__()
        
        # Create layers
        self.layers = nn.ModuleList([
            ErrorPropagationLayer(n1, h, n2)
            for n1, h, n2 in zip(
                layer_sizes[:-1],
                hidden_sizes,
                layer_sizes[1:]
            )
        ])
        
        # Initialize states
        self.layer_states = [
            torch.zeros(size)
            for size in layer_sizes
        ]
    
    def forward(self,
               input_data: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass through network.
        
        Args:
            input_data: Input tensor
            
        Returns:
            forward_errors: List of forward prediction errors
            backward_errors: List of backward prediction errors
        """
        # Bottom-up pass
        forward_errors = []
        current = input_data
        
        for i, layer in enumerate(self.layers):
            # Compute forward error
            forward_error, _ = layer(current)
            forward_errors.append(forward_error)
            
            # Update state and current input
            self.layer_states[i] = current
            current = forward_error
        
        # Top-down pass
        backward_errors = []
        
        for i in reversed(range(len(self.layers))):
            # Compute backward error
            _, backward_error = self.layers[i](
                self.layer_states[i],
                current if i < len(self.layers)-1 else None
            )
            backward_errors.append(backward_error)
            
            # Update current input
            current = backward_error
        
        return forward_errors, backward_errors
```

### Training Loop

```python
def train_network(network: ErrorPropagationNetwork,
                 dataset: torch.Tensor,
                 n_epochs: int = 100,
                 learning_rate: float = 0.01) -> List[float]:
    """Train error propagation network.
    
    Args:
        network: Error propagation network
        dataset: Training data
        n_epochs: Number of epochs
        learning_rate: Learning rate
        
    Returns:
        losses: Training losses
    """
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=learning_rate
    )
    losses = []
    
    for epoch in range(n_epochs):
        total_loss = 0
        
        for data in dataset:
            # Forward pass
            forward_errors, backward_errors = network(data)
            
            # Compute loss
            forward_loss = sum(
                torch.mean(error**2)
                for error in forward_errors
            )
            backward_loss = sum(
                torch.mean(error**2)
                for error in backward_errors
            )
            
            loss = forward_loss + backward_loss
            
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

### Network Design
1. Choose appropriate layer sizes
2. Design error functions
3. Initialize error scaling
4. Consider bidirectional flow

### Implementation
1. Monitor error propagation
2. Handle numerical stability
3. Validate predictions
4. Test error flow

### Training
1. Tune learning rates
2. Balance error components
3. Monitor error statistics
4. Validate learning

## Common Issues

### Technical Challenges
1. Error explosion
2. Gradient problems
3. State divergence
4. Learning collapse

### Solutions
1. Error normalization
2. Gradient clipping
3. State constraints
4. Careful initialization

## Related Documentation
- [[predictive_coding]]
- [[hierarchical_inference]]
- [[precision_weighting]] 