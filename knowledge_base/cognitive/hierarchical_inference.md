---
title: Hierarchical Inference
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
      - [[variational_inference]]
  - type: relates
    links:
      - [[free_energy_principle]]
      - [[active_inference]]
      - [[error_propagation]]
---

# Hierarchical Inference

## Overview

Hierarchical Inference is a framework for understanding how the brain processes information across multiple levels of abstraction. It suggests that cognitive processing occurs through a hierarchy of inference levels, where each level makes predictions about the level below and receives prediction errors from it.

## Core Concepts

### Hierarchical Generative Model
```math
p(x,h_1,...,h_L) = p(x|h_1)\prod_{l=1}^L p(h_l|h_{l+1})
```
where:
- $x$ is observation
- $h_l$ is hidden state at level $l$
- $L$ is number of levels

### Level-wise Inference
```math
q(h_l) \propto \exp(-F_l)
```
where:
- $q(h_l)$ is approximate posterior
- $F_l$ is level-specific free energy

## Implementation

### Hierarchical Layer

```python
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class HierarchicalLayer(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int):
        """Initialize hierarchical layer.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden dimension
            output_size: Output dimension
        """
        super().__init__()
        
        # Forward model
        self.forward_model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Backward model
        self.backward_model = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        
        # Precision parameters
        self.forward_precision = nn.Parameter(
            torch.ones(output_size)
        )
        self.backward_precision = nn.Parameter(
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
            forward_pred: Forward prediction
            backward_pred: Backward prediction
        """
        # Forward prediction
        forward_pred = self.forward_model(bottom_input)
        
        # Backward prediction
        if top_input is not None:
            backward_pred = self.backward_model(top_input)
        else:
            backward_pred = self.backward_model(forward_pred)
        
        return forward_pred, backward_pred
    
    def compute_errors(self,
                      bottom_input: torch.Tensor,
                      top_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute prediction errors.
        
        Args:
            bottom_input: Input from lower level
            top_input: Input from higher level
            
        Returns:
            forward_error: Forward prediction error
            backward_error: Backward prediction error
        """
        # Get predictions
        forward_pred, backward_pred = self.forward(
            bottom_input, top_input
        )
        
        # Compute errors
        forward_error = (
            (forward_pred - (top_input if top_input is not None else forward_pred)) *
            self.forward_precision
        )
        backward_error = (
            (bottom_input - backward_pred) *
            self.backward_precision
        )
        
        return forward_error, backward_error
```

### Hierarchical Network

```python
class HierarchicalNetwork(nn.Module):
    def __init__(self,
                 layer_sizes: List[int],
                 hidden_sizes: List[int]):
        """Initialize hierarchical network.
        
        Args:
            layer_sizes: List of layer sizes
            hidden_sizes: List of hidden sizes
        """
        super().__init__()
        
        # Create layers
        self.layers = nn.ModuleList([
            HierarchicalLayer(n1, h, n2)
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
            predictions: List of predictions
            errors: List of prediction errors
        """
        # Bottom-up pass
        forward_preds = []
        forward_errors = []
        current = input_data
        
        for i, layer in enumerate(self.layers):
            # Forward prediction
            pred, _ = layer(current)
            forward_preds.append(pred)
            
            # Compute error
            error, _ = layer.compute_errors(current)
            forward_errors.append(error)
            
            # Update state and current input
            self.layer_states[i] = current
            current = pred
        
        # Top-down pass
        backward_preds = []
        backward_errors = []
        
        for i in reversed(range(len(self.layers))):
            # Backward prediction
            _, pred = self.layers[i](
                self.layer_states[i],
                current if i < len(self.layers)-1 else None
            )
            backward_preds.append(pred)
            
            # Compute error
            _, error = self.layers[i].compute_errors(
                self.layer_states[i],
                current if i < len(self.layers)-1 else None
            )
            backward_errors.append(error)
            
            # Update current input
            current = pred
        
        return (forward_preds, backward_preds), (forward_errors, backward_errors)
```

### Training Loop

```python
def train_network(network: HierarchicalNetwork,
                 dataset: torch.Tensor,
                 n_epochs: int = 100,
                 learning_rate: float = 0.01) -> List[float]:
    """Train hierarchical network.
    
    Args:
        network: Hierarchical network
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
            predictions, errors = network(data)
            forward_errors, backward_errors = errors
            
            # Compute loss
            loss = sum(
                torch.mean(error**2)
                for error in forward_errors + backward_errors
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

### Network Design
1. Choose appropriate layer sizes
2. Design prediction functions
3. Initialize precisions
4. Consider depth vs. width

### Implementation
1. Monitor convergence
2. Handle numerical stability
3. Validate predictions
4. Test error propagation

### Training
1. Tune learning rates
2. Balance layer updates
3. Monitor error statistics
4. Validate learning

## Common Issues

### Technical Challenges
1. Vanishing gradients
2. Error instability
3. State divergence
4. Learning collapse

### Solutions
1. Skip connections
2. Layer normalization
3. Gradient clipping
4. Careful initialization

## Related Documentation
- [[predictive_coding]]
- [[variational_inference]]
- [[error_propagation]] 