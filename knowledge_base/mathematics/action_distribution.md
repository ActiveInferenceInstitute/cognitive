---
type: mathematical_concept
id: action_distribution_001
created: 2024-02-05
modified: 2024-02-05
tags: [active-inference, probability, action-selection, affordances]
aliases: [action-probabilities, policy-distribution]
---

# Action Probability Distribution

## Definition
At each time point, the action distribution represents the probability of selecting each available action (affordance), forming a proper probability distribution:

$E = [p(a_1), p(a_2), ..., p(a_n)]$ where $\sum_i p(a_i) = 1$

## Mathematical Structure

### Probability Vector
```python
class ActionDistribution:
    """Represents probability distribution over actions."""
    
    def __init__(self, num_actions: int):
        # Initialize uniform distribution
        self.probabilities = np.ones(num_actions) / num_actions
        
    def update(self, expected_free_energy: np.ndarray,
               temperature: float = 1.0) -> None:
        """Update action probabilities using softmax of -EFE."""
        self.probabilities = softmax(-temperature * expected_free_energy)
```

Links to:
- [[probability_simplex]]
- [[softmax_distribution]]
- [[categorical_distribution]]

### Properties
1. **Normalization**
   ```python
   def verify_normalization(probs: np.ndarray) -> bool:
       """Verify probability distribution properties."""
       return (np.all(probs >= 0) and 
               np.isclose(np.sum(probs), 1.0))
   ```
   Links to:
   - [[probability_axioms]]
   - [[measure_theory]]
   - [[normalization_constraints]]

2. **Entropy**
   ```python
   def compute_action_entropy(probs: np.ndarray) -> float:
       """Compute entropy of action distribution."""
       return -np.sum(probs * np.log(probs + 1e-10))
   ```
   Links to:
   - [[information_theory]]
   - [[exploration_exploitation]]
   - [[uncertainty_quantification]]

## Active Inference Context

### Free Energy Connection
The action probabilities are derived from the [[expected_free_energy]]:

$p(a) = \sigma(-\gamma G(a))$

where:
- $G(a)$ is the [[expected_free_energy]] for action $a$
- $\gamma$ is the [[precision_parameter]]
- $\sigma$ is the [[softmax_function]]

### Implementation Example
```python
class ActionSelector:
    """Select actions based on Expected Free Energy."""
    
    def __init__(self, num_actions: int, temperature: float = 1.0):
        self.num_actions = num_actions
        self.temperature = temperature
        self.distribution = np.ones(num_actions) / num_actions
    
    def update_distribution(self, 
                          expected_free_energy: np.ndarray) -> None:
        """Update action probabilities."""
        self.distribution = softmax(
            -self.temperature * expected_free_energy)
    
    def sample_action(self) -> int:
        """Sample action from current distribution."""
        return np.random.choice(
            self.num_actions, p=self.distribution)
```

Links to:
- [[policy_selection]]
- [[action_sampling]]
- [[stochastic_choice]]

## Temporal Aspects

### Single Time-Step
At each time $t$, the distribution represents immediate action probabilities:

$E_t = [p(a_1|s_t), p(a_2|s_t), ..., p(a_n|s_t)]$

Links to:
- [[markov_property]]
- [[conditional_probability]]
- [[temporal_dynamics]]

### Policy Relationship
The policy (sequence of actions) emerges from sequential application:

$\pi = (a_{t_1}, a_{t_2}, ..., a_{t_T})$ where each $a_{t_i} \sim E_{t_i}$

Links to:
- [[sequential_decision_making]]
- [[policy_composition]]
- [[temporal_planning]]

## Optimization

### Gradient-Based Updates
```python
def natural_gradient_update(distribution: np.ndarray,
                          gradient: np.ndarray,
                          learning_rate: float) -> np.ndarray:
    """Update distribution using natural gradient."""
    fisher = compute_fisher_matrix(distribution)
    natural_grad = np.linalg.solve(fisher, gradient)
    return softmax(np.log(distribution) + 
                  learning_rate * natural_grad)
```

Links to:
- [[natural_gradient]]
- [[fisher_information]]
- [[information_geometry]]

### Constraints
1. **Probability Constraints**
   - Non-negativity: $p(a_i) \geq 0$
   - Normalization: $\sum_i p(a_i) = 1$
   - Links to [[constraint_optimization]]

2. **Exploration Control**
   - Temperature parameter $\gamma$
   - Entropy regularization
   - Links to [[exploration_strategies]]

## Applications

### Decision Making
- [[action_selection]]
- [[behavioral_control]]
- [[motor_planning]]

### Learning
- [[policy_learning]]
- [[reinforcement_learning]]
- [[adaptive_behavior]]

### Analysis
- [[behavioral_analysis]]
- [[decision_theory]]
- [[information_theory]]

## References
- [[friston_2017]] - Active Inference and Learning
- [[sutton_2018]] - Reinforcement Learning
- [[amari_2000]] - Information Geometry 