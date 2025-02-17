---
type: matrix_spec
id: A_matrix_001
matrix_type: perception
created: {{date}}
modified: {{date}}
tags: [matrix, perception, active-inference]
related_spaces: [o_space, s_space]
---

# A-Matrix (Perception/Likelihood)

## Definition
The A-matrix defines the mapping between hidden states ([[s_space]]) and observations ([[o_space]]), representing the likelihood P(o|s).

## Matrix Structure
```yaml
dimensions:
  rows: num_observations    # From [[o_space]]
  cols: num_states         # From [[s_space]]
shape_constraints:
  - rows > 0
  - cols > 0
  - sum(cols) == 1.0  # Each column sums to 1 (probability distribution)
```

## Mathematical Form
```python
# Likelihood mapping
P(o|s) = A[o, s]

# Constraints
∀s: ∑_o A[o,s] = 1  # Column-wise normalization
A[o,s] ≥ 0          # Non-negative probabilities
```

## Data Structure
```yaml
matrix_data:
  format: numpy.ndarray
  dtype: float32
  initialization: random_stochastic  # Random initialization preserving constraints
  storage: matrix_store/A_matrix.npy
```

## Visualization
```yaml
plot_type: heatmap
colormap: "YlOrRd"
title: "Perception Matrix (A)"
xlabel: "Hidden States"
ylabel: "Observations"
```

## Update Rules
- [[gradient_A]] - Gradient updates for learning
- [[dirichlet_A]] - Dirichlet prior updates
- [[empirical_A]] - Learning from data

## Properties
- Represents sensory mapping
- Columns are probability distributions
- Sparse for efficient perception
- [[precision_A]] defines confidence

## Integration
- Used in [[belief_updating]]
- Affects [[free_energy_computation]]
- Links to [[observation_model]]

## Examples
```python
# Example 3x4 A-matrix
A = [
    [0.7, 0.2, 0.1, 0.0],
    [0.2, 0.7, 0.2, 0.3],
    [0.1, 0.1, 0.7, 0.7]
]
```

## Related Components
- [[o_space]] - Observation space definition
- [[s_space]] - State space definition
- [[precision_parameters]] - Precision hyperparameters
- [[likelihood_learning]] - Learning algorithms

## References
- [[active_inference_theory]]
- [[matrix_operations]]
- [[probability_theory]] 