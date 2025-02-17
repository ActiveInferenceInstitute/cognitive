---
title: Probability Theory
type: concept
status: stable
created: 2024-02-12
tags:
  - mathematics
  - probability
  - foundations
semantic_relations:
  - type: extends
    links: [[measure_theory]]
  - type: relates
    links:
      - [[information_theory]]
      - [[statistics]]
---

# Probability Theory

## Overview

Probability theory provides the mathematical foundation for reasoning under uncertainty, forming the basis for modern approaches to cognitive modeling, machine learning, and statistical inference.

## Fundamentals

### Probability Spaces

#### Measure Space
```math
(\Omega, \mathcal{F}, P)
```
where:
- $\Omega$ is sample space
- $\mathcal{F}$ is σ-algebra
- $P$ is probability measure

#### Axioms
1. Non-negativity: $P(A) \geq 0$
2. Normalization: $P(\Omega) = 1$
3. Additivity: $P(\cup_i A_i) = \sum_i P(A_i)$ for disjoint sets

### Random Variables

#### Definition
A measurable function $X: \Omega \rightarrow \mathbb{R}$

#### Distribution Function
```math
F_X(x) = P(X \leq x)
```

#### Density Function
```math
f_X(x) = \frac{d}{dx}F_X(x)
```

## Key Concepts

### Conditional Probability
```math
P(A|B) = \frac{P(A \cap B)}{P(B)}
```

### Bayes' Theorem
```math
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
```

### Independence
```math
P(A \cap B) = P(A)P(B)
```

## Common Distributions

### Gaussian Distribution
```python
class GaussianDistribution:
    def __init__(self, mu: float, sigma: float):
        """Initialize Gaussian distribution.
        
        Args:
            mu: Mean
            sigma: Standard deviation
        """
        self.mu = mu
        self.sigma = sigma
    
    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        """Compute probability density."""
        z = (x - self.mu) / self.sigma
        return torch.exp(-0.5 * z**2) / (self.sigma * math.sqrt(2 * math.pi))
    
    def sample(self, n: int) -> torch.Tensor:
        """Generate samples."""
        return torch.randn(n) * self.sigma + self.mu
```

### Categorical Distribution
```python
class CategoricalDistribution:
    def __init__(self, probs: torch.Tensor):
        """Initialize categorical distribution.
        
        Args:
            probs: Probability vector
        """
        self.probs = probs
        
    def pmf(self, x: torch.Tensor) -> torch.Tensor:
        """Compute probability mass."""
        return self.probs[x]
    
    def sample(self, n: int) -> torch.Tensor:
        """Generate samples."""
        return torch.multinomial(self.probs, n, replacement=True)
```

## Applications

### Probabilistic Modeling
```python
class ProbabilisticModel:
    def __init__(self, prior: Distribution):
        """Initialize probabilistic model.
        
        Args:
            prior: Prior distribution
        """
        self.prior = prior
        
    def likelihood(self,
                  x: torch.Tensor,
                  theta: torch.Tensor) -> torch.Tensor:
        """Compute likelihood.
        
        Args:
            x: Observations
            theta: Parameters
            
        Returns:
            likelihood: p(x|theta)
        """
        raise NotImplementedError
    
    def posterior(self,
                 x: torch.Tensor,
                 theta: torch.Tensor) -> torch.Tensor:
        """Compute posterior.
        
        Args:
            x: Observations
            theta: Parameters
            
        Returns:
            posterior: p(theta|x)
        """
        likelihood = self.likelihood(x, theta)
        prior = self.prior.pdf(theta)
        return likelihood * prior
```

### Inference Methods
```python
class BayesianInference:
    def __init__(self, model: ProbabilisticModel):
        """Initialize Bayesian inference.
        
        Args:
            model: Probabilistic model
        """
        self.model = model
    
    def map_estimate(self,
                    x: torch.Tensor,
                    init_theta: torch.Tensor,
                    num_steps: int = 1000) -> torch.Tensor:
        """Compute MAP estimate.
        
        Args:
            x: Observations
            init_theta: Initial parameters
            num_steps: Number of optimization steps
            
        Returns:
            theta_map: MAP estimate
        """
        theta = init_theta.requires_grad_()
        optimizer = torch.optim.Adam([theta])
        
        for _ in range(num_steps):
            # Compute negative log posterior
            neg_log_posterior = -torch.log(
                self.model.posterior(x, theta)
            )
            
            # Update parameters
            optimizer.zero_grad()
            neg_log_posterior.backward()
            optimizer.step()
        
        return theta.detach()
```

## Advanced Topics

### Information Theory
```python
def entropy(p: torch.Tensor) -> torch.Tensor:
    """Compute entropy.
    
    Args:
        p: Probability distribution
        
    Returns:
        H: Entropy value
    """
    return -torch.sum(p * torch.log(p))

def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence.
    
    Args:
        p: First distribution
        q: Second distribution
        
    Returns:
        KL: KL divergence
    """
    return torch.sum(p * torch.log(p / q))
```

### Exponential Family
```python
class ExponentialFamily:
    def __init__(self,
                 natural_params: torch.Tensor,
                 sufficient_stats: Callable):
        """Initialize exponential family distribution.
        
        Args:
            natural_params: Natural parameters
            sufficient_stats: Sufficient statistics function
        """
        self.eta = natural_params
        self.T = sufficient_stats
    
    def log_partition(self) -> torch.Tensor:
        """Compute log partition function."""
        raise NotImplementedError
    
    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        """Compute probability density."""
        return torch.exp(
            torch.sum(self.eta * self.T(x)) - self.log_partition()
        )
```

## Best Practices

### Implementation
1. Use log-space computations
2. Implement numerical stability
3. Validate probability axioms
4. Handle edge cases

### Modeling
1. Choose appropriate distributions
2. Validate assumptions
3. Consider conjugate priors
4. Test inference methods

### Computation
1. Use stable algorithms
2. Implement vectorization
3. Handle numerical precision
4. Validate results

## Common Issues

### Numerical Stability
1. Underflow/overflow
2. Division by zero
3. Log of zero
4. Precision loss

### Solutions
1. Log-space arithmetic
2. Stable algorithms
3. Numerical safeguards
4. Error checking

## Related Documentation
- [[measure_theory]]
- [[information_theory]]
- [[statistics]] 