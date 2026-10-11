---
type: matrix_spec
id: D_matrix_001
matrix_type: prior
created: 2025-01-01
modified: 2025-01-01
tags:
- matrix
- prior
- active-inference
related_spaces:
- s_space
---


# D-Matrix (Prior Beliefs)

## Definition

The D-matrix encodes prior beliefs about hidden states, representing P(s₁) for the initial state and default beliefs.

## Matrix Structure

```yaml

dimensions:

  size: num_states    # From [[s_space]]

shape_constraints:

  - sum(values) == 1.0  # Probability distribution

  - all_values >= 0     # Non-negative probabilities

```

## Mathematical Form

See `docs/examples/README.md` for runnable examples.


## Data Structure

```yaml

matrix_data:

  format: numpy.ndarray

  dtype: float32

  shape: [num_states]

  initialization: uniform  # Start with uniform prior

  storage: matrix_store/D_matrix.npy

```

## Visualization

```yaml

plot_type: bar

colormap: "Purples"

title: "Prior Beliefs (D)"

xlabel: "Hidden States"

ylabel: "Probability"

```

## Properties

### Belief Structure

- Initial state distribution

- Default/reset beliefs

- [[belief_initialization]]

### Information Content

- Prior knowledge encoding

- [[information_theory]]

- entropy analysis

### Learning Effects

- Prior adaptation

- Experience integration

- [[belief_updating]]

## Update Rules

- [[empirical_prior]] - Prior estimated from observed data

- hierarchical prior - Multi-level priors

- context prior - Context-dependent priors

## Integration

- Initializes [[belief_state]]

- Affects free energy computation

- Guides [[exploration_exploitation]]

## Examples

```python

# Uniform prior over 4 states

D_uniform = np.array([0.25, 0.25, 0.25, 0.25])

# Informed prior (bias towards state 2)

D_informed = np.array([0.1, 0.7, 0.1, 0.1])

# Highly certain prior

D_certain = np.array([0.95, 0.02, 0.02, 0.01])

```

## Computational Interface

```python

class PriorBeliefs:

    def __init__(self, num_states: int):

        self.D = self._initialize_prior(num_states)

    def get_prior(self) -> np.ndarray:

        """Get prior belief distribution"""

        return self.D.copy()

    def update_prior(self,

                    empirical_dist: np.ndarray,

                    learning_rate: float):

        """Update prior based on empirical distribution"""

        pass

    def compute_kl_divergence(self,

                            belief: np.ndarray) -> float:

        """Compute KL divergence from prior"""

        pass

```

## Learning Methods

- [[bayesian_updating]] - Posterior computation

- prior learning - Adapting priors

- [[structure_learning]] - Learning belief structure

## Applications

### Belief Initialization

- cold start - Starting inference with no prior experience

- belief reset - Restoring prior beliefs after a surprising event

- default behavior - Fallback behavior under uninformative priors

### Information Processing

- surprise computation

- novelty detection

- [[anomaly_detection]]

## Related Components

- [[s_space]] - Set of hidden states the prior covers

- [[belief_state]] - Current beliefs

- [[information_gain]] - Epistemic value

- [[prior_knowledge]] - Background knowledge expressed as a prior

## References

- [[bayesian_inference]]

- [[information_theory]]

- [[belief_systems]]

