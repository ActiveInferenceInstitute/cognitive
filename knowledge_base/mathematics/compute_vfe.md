---
type: implementation
id: compute_vfe_001
created: 2024-02-05
modified: 2024-02-05
tags: [active-inference, free-energy, implementation]
aliases: [compute-vfe, variational-free-energy]
---

# Computing Variational Free Energy

## Mathematical Definition

The Variational Free Energy (VFE) is defined as:

$F = \mathbb{E}_{Q(s)}[\ln Q(s) - \ln P(o,s)]$

Which decomposes into:

$F = \underbrace{-\mathbb{E}_{Q(s)}[\ln P(o|s)]}_{\text{Accuracy}} + \underbrace{D_{KL}[Q(s)\|P(s)]}_{\text{Complexity}}$

## Implementation

```python
def compute_variational_free_energy(
    observation: int,        # Current observation
    beliefs: np.ndarray,     # Current belief distribution Q(s)
    prior: np.ndarray,       # Prior belief distribution P(s)
    A: np.ndarray,           # Observation model P(o|s)
    return_components: bool = False
) -> Union[float, Tuple[float, float, float]]:
    """Compute Variational Free Energy.
    
    Args:
        observation: Observed state index
        beliefs: Current belief distribution Q(s)
        prior: Prior belief distribution P(s)
        A: Observation likelihood matrix P(o|s)
        return_components: Whether to return accuracy and complexity terms
        
    Returns:
        If return_components is False:
            Total Variational Free Energy
        If return_components is True:
            Tuple of (total_VFE, accuracy, complexity)
    """
    # Compute accuracy term (negative log likelihood)
    likelihood = A[observation, :]
    accuracy = -np.sum(beliefs * np.log(likelihood + 1e-12))
    
    # Compute complexity term (KL from prior)
    complexity = kl_divergence(beliefs, prior)
    
    # Total Variational Free Energy
    total_vfe = accuracy + complexity
    
    if return_components:
        return total_vfe, accuracy, complexity
    return total_vfe
```

## Components

### Accuracy Term
- Measures how well beliefs explain observations
- Negative log likelihood of data
- Drives perceptual accuracy
- Links to [[prediction_error]]

### Complexity Term
- Measures divergence from prior beliefs
- KL divergence from prior
- Penalizes complex explanations
- Links to [[bayesian_complexity]]

## Usage

### In Belief Updating
```python
def update_beliefs(
    observation: int,
    prior: np.ndarray,
    A: np.ndarray,
    learning_rate: float = 0.1
) -> np.ndarray:
    """Update beliefs using VFE minimization."""
    # Compute likelihood
    likelihood = A[observation, :]
    
    # Compute posterior using Bayes rule
    posterior = likelihood * prior
    posterior /= posterior.sum()
    
    # Compute VFE for monitoring
    vfe, acc, comp = compute_variational_free_energy(
        observation=observation,
        beliefs=posterior,
        prior=prior,
        A=A,
        return_components=True
    )
    
    return posterior, vfe, acc, comp
```

### In Monitoring
```python
def monitor_inference(model, observation: int):
    """Monitor inference quality using VFE."""
    vfe, acc, comp = compute_variational_free_energy(
        observation=observation,
        beliefs=model.beliefs,
        prior=model.prior,
        A=model.A,
        return_components=True
    )
    
    print(f"VFE: {vfe:.3f}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Complexity: {comp:.3f}")
```

## Properties

### Mathematical Properties
- Upper bounds surprise: $F \geq -\ln P(o)$
- Non-negative complexity term
- Convex in Q(s)
- Links to [[variational_inference]]

### Computational Properties
- O(n) complexity for n states
- Numerically stable with log space
- Parallelizable across samples
- Links to [[numerical_methods]]

## Visualization

### Key Plots
- [[vfe_landscape]]: VFE surface over beliefs
- [[accuracy_complexity]]: Trade-off visualization
- [[convergence_plot]]: VFE during inference

## Related Implementations
- [[compute_efe]]: Expected Free Energy
- [[update_beliefs]]: Belief updating
- [[monitor_inference]]: Inference quality

## References
- [[friston_2006]] - Free Energy Principle
- [[bogacz_2017]] - Tutorial Paper
- [[buckley_2017]] - Active Inference Tutorial 