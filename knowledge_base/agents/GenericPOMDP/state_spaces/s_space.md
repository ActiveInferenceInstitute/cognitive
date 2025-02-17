---
type: state_space
id: s_space_001
space_type: hidden_state
created: {{date}}
modified: {{date}}
tags: [state-space, hidden-states, active-inference]
related_matrices: [A_matrix, B_matrix, D_matrix]
---

# Hidden State Space (S)

## Definition
The hidden state space defines the possible internal states of the generative model that the agent infers through active inference.

## Space Structure
```yaml
dimensions:
  size: num_states
  factors: []  # Optional factorization for structured state spaces
constraints:
  - finite
  - discrete
  - positive_entropy
```

## State Enumeration
```yaml
states:
  format: categorical
  encoding: one_hot
  values:
    - state_1:
        id: 0
        description: "First state"
        prior_probability: 0.25
    - state_2:
        id: 1
        description: "Second state"
        prior_probability: 0.25
    - state_3:
        id: 2
        description: "Third state"
        prior_probability: 0.25
    - state_4:
        id: 3
        description: "Fourth state"
        prior_probability: 0.25
```

## Belief Representation
```python
# Belief vector over states
s = [P(s_1), P(s_2), ..., P(s_n)]

# Constraints
sum(s) == 1.0  # Probability distribution
s[i] >= 0      # Non-negative probabilities
```

## Integration Points
- [[A_matrix]] - Observation generation
- [[B_matrix]] - State transitions
- [[D_matrix]] - Prior beliefs
- [[state_estimation]] - Inference process

## Properties
### Markov Property
- Current state contains all relevant information
- [[markov_assumptions]]
- [[temporal_dependencies]]

### Observability
- Partially observable through [[A_matrix]]
- [[information_gain]] calculations
- [[uncertainty_resolution]]

## State Dynamics
- [[transition_rules]]
- [[state_constraints]]
- [[boundary_conditions]]

## Visualization
```yaml
visualization:
  type: state_diagram
  layout: force_directed
  node_color: state_probability
  edge_color: transition_probability
```

## Analysis Tools
- [[state_entropy]] - Uncertainty measure
- [[state_distance]] - Metric between states
- [[state_clustering]] - State space structure

## Learning
- [[state_discovery]] - Identifying new states
- [[state_pruning]] - Removing redundant states
- [[state_refinement]] - Improving state representations

## Computational Interface
```python
class StateSpace:
    def __init__(self, num_states: int):
        self.num_states = num_states
        self.current_belief = np.ones(num_states) / num_states
    
    def update_belief(self, evidence: np.ndarray) -> np.ndarray:
        """Update belief state with new evidence"""
        pass
    
    def entropy(self) -> float:
        """Calculate belief uncertainty"""
        pass
    
    def sample(self) -> int:
        """Sample a state from current belief"""
        pass
```

## Related Components
- [[belief_updating]] - Inference methods
- [[policy_selection]] - Action selection
- [[information_gain]] - Exploration value
- [[state_abstraction]] - Hierarchical structure

## References
- [[state_space_theory]]
- [[belief_representations]]
- [[information_theory]] 