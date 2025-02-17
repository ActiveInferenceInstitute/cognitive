---
title: Variational Inference
type: concept
status: stable
created: 2024-02-12
tags:
  - mathematics
  - probability
  - computation
semantic_relations:
  - type: foundation
    links: 
      - [[probability_theory]]
      - [[information_theory]]
  - type: relates
    links:
      - [[free_energy_principle]]
      - [[predictive_coding]]
      - [[optimization_theory]]
---

# Variational Inference

## Overview

Variational Inference (VI) is a method for approximating complex probability distributions through optimization. It forms the mathematical foundation for many cognitive modeling approaches, including predictive coding and active inference.

## Core Concepts

### Evidence Lower Bound
```math
\mathcal{L} = \mathbb{E}_{q(z)}[\log p(x,z) - \log q(z)]
```
where:
- $\mathcal{L}$ is ELBO
- $q(z)$ is variational distribution
- $p(x,z)$ is joint distribution

### KL Divergence
```math
D_{KL}(q||p) = \mathbb{E}_{q(z)}[\log q(z) - \log p(z)]
```
where:
- $D_{KL}$ is KL divergence
- $q(z)$ is approximate posterior
- $p(z)$ is true posterior

## Implementation

### Variational Distribution

```python
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from torch.distributions import Normal, kl_divergence

class VariationalDistribution(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 latent_size: int):
        """Initialize variational distribution.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden dimension
            latent_size: Latent dimension
        """
        super().__init__()
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Distribution parameters
        self.mean = nn.Linear(hidden_size, latent_size)
        self.log_var = nn.Linear(hidden_size, latent_size)
    
    def forward(self,
               x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through distribution.
        
        Args:
            x: Input tensor
            
        Returns:
            z: Sampled latent
            mean: Distribution mean
            log_var: Log variance
        """
        # Encode input
        h = self.encoder(x)
        
        # Get distribution parameters
        mean = self.mean(h)
        log_var = self.log_var(h)
        
        # Sample using reparameterization
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        return z, mean, log_var
```

### Generative Model

```python
class GenerativeModel(nn.Module):
    def __init__(self,
                 latent_size: int,
                 hidden_size: int,
                 output_size: int):
        """Initialize generative model.
        
        Args:
            latent_size: Latent dimension
            hidden_size: Hidden dimension
            output_size: Output dimension
        """
        super().__init__()
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self,
               z: torch.Tensor) -> torch.Tensor:
        """Forward pass through model.
        
        Args:
            z: Latent tensor
            
        Returns:
            x_hat: Reconstructed input
        """
        return self.decoder(z)
```

### Variational Autoencoder

```python
class VariationalAutoencoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 latent_size: int):
        """Initialize variational autoencoder.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden dimension
            latent_size: Latent dimension
        """
        super().__init__()
        
        # Model components
        self.encoder = VariationalDistribution(
            input_size, hidden_size, latent_size
        )
        self.decoder = GenerativeModel(
            latent_size, hidden_size, input_size
        )
    
    def forward(self,
               x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through model.
        
        Args:
            x: Input tensor
            
        Returns:
            x_hat: Reconstructed input
            mean: Latent mean
            log_var: Latent log variance
        """
        # Encode input
        z, mean, log_var = self.encoder(x)
        
        # Decode latent
        x_hat = self.decoder(z)
        
        return x_hat, mean, log_var
    
    def loss_function(self,
                     x: torch.Tensor,
                     x_hat: torch.Tensor,
                     mean: torch.Tensor,
                     log_var: torch.Tensor) -> torch.Tensor:
        """Compute ELBO loss.
        
        Args:
            x: Input tensor
            x_hat: Reconstructed input
            mean: Latent mean
            log_var: Latent log variance
            
        Returns:
            loss: ELBO loss
        """
        # Reconstruction loss
        recon_loss = torch.mean(
            torch.sum((x - x_hat)**2, dim=1)
        )
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(
            torch.sum(
                1 + log_var - mean**2 - torch.exp(log_var),
                dim=1
            )
        )
        
        return recon_loss + kl_loss
```

### Training Loop

```python
def train_model(model: VariationalAutoencoder,
                dataset: torch.Tensor,
                n_epochs: int = 100,
                learning_rate: float = 0.001) -> List[float]:
    """Train variational autoencoder.
    
    Args:
        model: Variational autoencoder
        dataset: Training data
        n_epochs: Number of epochs
        learning_rate: Learning rate
        
    Returns:
        losses: Training losses
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
    losses = []
    
    for epoch in range(n_epochs):
        total_loss = 0
        
        for data in dataset:
            # Forward pass
            x_hat, mean, log_var = model(data)
            
            # Compute loss
            loss = model.loss_function(
                data, x_hat, mean, log_var
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
1. Choose appropriate architectures
2. Design latent space
3. Initialize parameters
4. Consider hierarchical structure

### Implementation
1. Monitor convergence
2. Handle numerical stability
3. Validate inference
4. Test reconstruction

### Training
1. Tune learning rates
2. Balance loss components
3. Monitor KL divergence
4. Validate learning

## Common Issues

### Technical Challenges
1. Posterior collapse
2. Latent space issues
3. Gradient problems
4. Training instability

### Solutions
1. KL annealing
2. Warm-up period
3. Gradient clipping
4. Careful initialization

## Related Documentation
- [[probability_theory]]
- [[information_theory]]
- [[optimization_theory]]
