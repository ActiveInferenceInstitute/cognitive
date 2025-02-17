---
type: matrix_spec
id: E_matrix_001
matrix_type: action_distribution
created: 2024-02-05
modified: 2024-02-05
tags: [matrix, action-distribution, active-inference]
related_spaces: [action_space]
---

# E-Matrix (Action Distribution)

## Definition
The E-matrix represents the probability distribution over available actions at each time point, encoding the likelihood of selecting each possible action (affordance). This distribution is derived from the [[expected_free_energy]] of each action.

## Matrix Structure
```yaml
dimensions:
  size: num_actions  # Number of available actions
shape_constraints:
  - sum(probabilities) == 1.0  # Probability distribution
  - all_values >= 0  # Non-negative probabilities
```

## Mathematical Form
$E = [p(a_1), p(a_2), ..., p(a_n)]$ where $p(a_i) = \sigma(-\gamma G(a_i))$

where:
- $p(a_i)$ is the probability of selecting action $i$
- $G(a_i)$ is the [[expected_free_energy]] for action $i$
- $\gamma$ is the [[precision_parameter]] (inverse temperature)
- $\sigma$ is the [[softmax_function]]
- $\sum_i p(a_i) = 1$ (normalization constraint)

Links to:
- [[probability_distribution]]
- [[action_selection]]
- [[policy_optimization]]

## Data Structure
```yaml
matrix_data:
  format: numpy.ndarray
  dtype: float32  # Probability values
  shape: [num_actions]  # Vector of action probabilities
  initialization: uniform  # Initially uniform distribution
  storage: matrix_store/E_matrix.npy
```

## Properties

### Probability Properties
- Normalized distribution: $\sum_i p(a_i) = 1$
- Non-negative values: $p(a_i) \geq 0$
- Links to [[probability_axioms]]

### Selection Mechanism
```python
def sample_action(E: np.ndarray) -> int:
    """Sample action from probability distribution."""
    return np.random.choice(len(E), p=E)
```
Links to:
- [[stochastic_sampling]]
- [[categorical_distribution]]
- [[action_selection]]

## Update Rules

### Free Energy Based
```python
def update_distribution(
    expected_free_energy: np.ndarray,
    temperature: float = 1.0
) -> np.ndarray:
    """Update action probabilities based on Expected Free Energy."""
    return softmax(-temperature * expected_free_energy)
```

Links to:
- [[expected_free_energy]]
- [[softmax_function]]
- [[temperature_parameter]]

### Validation
```python
def validate_distribution(E: np.ndarray) -> bool:
    """Validate probability distribution properties."""
    return (
        np.allclose(np.sum(E), 1.0) and  # Normalized
        np.all(E >= 0) and               # Non-negative
        len(E.shape) == 1                # Vector form
    )
```

### Learning Updates
- [[policy_learning]] - Adapting action probabilities
- [[exploration_exploitation]] - Balancing exploration/exploitation
- [[preference_learning]] - Learning from outcomes

## Integration

### Active Inference Process
1. Compute [[expected_free_energy]] for each action
2. Update E-matrix using softmax transformation
3. Sample action from resulting distribution

### System Components
- Input from [[expected_free_energy]]
- Affects [[action_selection]]
- Guides [[behavioral_control]]

## Visualization

### Distribution Plot
```python
def plot_action_distribution(E: np.ndarray,
                           action_labels: List[str] = None):
    """Plot action probability distribution."""
    plt.figure(figsize=(10, 5))
    sns.barplot(x=range(len(E)), y=E)
    plt.title('Action Probability Distribution')
    plt.xlabel('Actions')
    plt.ylabel('Probability')
    if action_labels:
        plt.xticks(range(len(E)), action_labels)
    plt.ylim(0, 1)
```

### Temperature Effects
```python
def plot_temperature_effects(G: np.ndarray,
                           temperatures: List[float]):
    """Visualize effect of temperature on action distribution."""
    plt.figure(figsize=(12, 6))
    for temp in temperatures:
        E = softmax(-temp * G)
        plt.plot(E, label=f'T={temp}')
    plt.title('Temperature Effects on Action Distribution')
    plt.xlabel('Actions')
    plt.ylabel('Probability')
    plt.legend()
```

## Examples

### Basic Distribution
```python
# Example action distribution for 3 actions
E = np.array([0.4, 0.4, 0.2])  # Must sum to 1.0
```

### With Temperature
```python
def compute_distribution(G: np.ndarray, temp: float) -> np.ndarray:
    """Compute action distribution with temperature."""
    return softmax(-temp * G)

# Different exploration regimes
E_explore = compute_distribution(G, temp=0.1)  # High exploration
E_exploit = compute_distribution(G, temp=10.0)  # High exploitation
```

## Computational Interface
```python
class ActionDistribution:
    """Manages action probability distribution."""
    
    def __init__(self, num_actions: int):
        """Initialize uniform distribution."""
        self.E = np.ones(num_actions) / num_actions
        
    def update(self, expected_free_energy: np.ndarray,
               temperature: float) -> None:
        """Update distribution based on Expected Free Energy."""
        self.E = softmax(-temperature * expected_free_energy)
        assert validate_distribution(self.E), "Invalid distribution"
        
    def sample(self) -> int:
        """Sample action from distribution."""
        return np.random.choice(len(self.E), p=self.E)
        
    def entropy(self) -> float:
        """Compute distribution entropy."""
        return -np.sum(self.E * np.log(self.E + 1e-10))
```

## Applications

### Decision Making
- [[action_selection]] - Selecting actions
- [[exploration_strategies]] - Managing exploration
- [[behavioral_control]] - Controlling behavior

### Analysis
- [[entropy_analysis]] - Distribution uncertainty
- [[information_gain]] - Information theoretic measures
- [[decision_metrics]] - Performance evaluation

## Related Components
- [[action_space]] - Space of possible actions
- [[expected_free_energy]] - Drives distribution updates
- [[policy_selection]] - Higher-level selection
- [[behavioral_modeling]] - Behavioral analysis

## References
- [[friston_2017]] - Active Inference and Learning
- [[sutton_2018]] - Reinforcement Learning
- [[amari_2000]] - Information Geometry 