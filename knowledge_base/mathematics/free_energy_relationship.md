---
type: concept
id: free_energy_relationship_001
created: 2024-02-05
modified: 2024-02-05
tags: [active-inference, free-energy, theory]
aliases: [vfe-efe-relationship, free-energy-duality]
---

# Relationship Between VFE and EFE

## Overview

The relationship between Variational Free Energy (VFE) and Expected Free Energy (EFE) is fundamental to understanding Active Inference. While VFE quantifies current model fit, EFE guides future actions through prediction.

## Mathematical Connection

### Present vs Future
- VFE: $F = \mathbb{E}_{Q(x)}[-\ln P(y|x)] + D_{KL}[Q(x)\|P(x)]$
- EFE: $G(\pi) = -\mathbb{E}_{Q(\tilde{x},\tilde{y}|\pi)}[D_{KL}[Q(\tilde{x}|\tilde{y},\pi)\|Q(\tilde{x}|\pi)]] - \mathbb{E}_{Q(\tilde{y}|\pi)}[\ln P(\tilde{y}|C)]$

### Key Differences
1. Temporal Scope
   - VFE: Current state estimation
   - EFE: Future state prediction
   
2. Optimization Target
   - VFE: Minimize perception error
   - EFE: Optimize action selection

3. Component Focus
   - VFE: Accuracy vs Complexity
   - EFE: Epistemic vs Pragmatic value

## Implementation Details

```python
def compute_free_energies(
    model,
    observation: np.ndarray,
    action: Optional[int] = None
) -> Tuple[float, float]:
    """Compute both VFE and EFE for comparison.
    
    Args:
        model: Active Inference model instance
        observation: Current observation
        action: Optional action for EFE computation
        
    Returns:
        Tuple of (VFE, EFE) values
    """
    # Compute VFE
    vfe = model.compute_vfe(
        observation=observation,
        return_components=False
    )
    
    # Compute EFE if action is provided
    efe = None
    if action is not None:
        efe = model.compute_expected_free_energy(
            action_idx=action,
            return_components=False
        )
    
    return vfe, efe

def analyze_free_energy_relationship(
    model,
    time_window: int = 20
) -> Dict[str, np.ndarray]:
    """Analyze relationship between VFE and EFE over time.
    
    Args:
        model: Active Inference model instance
        time_window: Number of time steps to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    results = {
        'time_steps': np.arange(time_window),
        'vfe_values': np.zeros(time_window),
        'efe_values': np.zeros(time_window),
        'correlation': np.zeros(time_window-1),
        'prediction_error': np.zeros(time_window-1)
    }
    
    # Simulate and collect data
    for t in range(time_window):
        # Get current state
        observation = model.get_observation()
        action = model.select_action()
        
        # Compute free energies
        vfe, efe = compute_free_energies(
            model=model,
            observation=observation,
            action=action
        )
        
        # Store values
        results['vfe_values'][t] = vfe
        results['efe_values'][t] = efe
        
        # Update model
        model.step(action)
        
        # Compute relationships for t > 0
        if t > 0:
            # Correlation between VFE and EFE
            results['correlation'][t-1] = np.corrcoef(
                results['vfe_values'][:t],
                results['efe_values'][:t]
            )[0,1]
            
            # Prediction error (how well EFE predicted next VFE)
            results['prediction_error'][t-1] = np.abs(
                results['efe_values'][t-1] - results['vfe_values'][t]
            )
    
    return results
```

## Key Properties

### 1. Temporal Dependency
- VFE depends on current observations
- EFE depends on predicted future states
- Both contribute to belief updating

### 2. Information Flow
- VFE → Belief Update → Action Selection
- EFE → Policy Selection → Action Execution
- Circular causation through action-perception cycle

### 3. Optimization Characteristics
- VFE: Convex optimization
- EFE: Non-convex optimization
- Different convergence properties

## Practical Implications

### 1. Model Design
- Balance between components
- Proper scaling of terms
- Numerical stability

### 2. Algorithm Implementation
- Sequential computation
- Memory requirements
- Computational efficiency

### 3. Performance Analysis
- Convergence metrics
- Behavioral patterns
- Learning dynamics

## Related Concepts
- [[belief_updating]]
- [[policy_selection]]
- [[active_inference_cycle]]
- [[optimization_methods]]

## Common Challenges

### 1. Numerical Issues
- Scale differences
- Gradient computation
- Stability concerns

### 2. Implementation Complexity
- Component balance
- Parameter tuning
- Convergence monitoring

### 3. Analysis Difficulties
- Interpretation of values
- Component attribution
- Performance assessment

## Best Practices

### 1. Implementation
- Use stable numerical methods
- Monitor component ratios
- Implement sanity checks

### 2. Analysis
- Track both measures
- Compare trajectories
- Validate predictions

### 3. Optimization
- Balance update rates
- Monitor convergence
- Validate results

## References
- [[friston_2015]] - Active Inference Theory
- [[parr_2019]] - Relationship Analysis
- [[da_costa_2020]] - Computational Implementation 