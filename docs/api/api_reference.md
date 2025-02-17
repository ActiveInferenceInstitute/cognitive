# API Reference Documentation

---
title: API Reference Documentation
type: reference
status: stable
created: 2024-02-06
tags:
  - api
  - reference
  - documentation
semantic_relations:
  - type: implements
    links: [[api_documentation]]
  - type: documents
    links: 
      - [[knowledge_base/cognitive/active_inference|Active Inference Theory]]
      - [[knowledge_base/cognitive/free_energy_principle|Free Energy Principle]]
  - type: relates
    links:
      - [[package_documentation]]
      - [[documentation_standards]]
---

## Core API Components

### Active Inference Framework
See [[knowledge_base/cognitive/active_inference|Active Inference Theory]] for theoretical background.

#### ActiveInferenceAgent
```python
class ActiveInferenceAgent:
    """
    Main agent implementation for Active Inference.
    
    Theory: [[knowledge_base/cognitive/active_inference|Active Inference]]
    Examples: [[active_inference_example]]
    """
    
    def __init__(self, 
                 observation_space: Space,
                 action_space: Space,
                 precision: float = 1.0):
        """
        Initialize agent with observation and action spaces.
        
        Theory: [[knowledge_base/cognitive/free_energy_principle|Free Energy Principle]]
        
        Args:
            observation_space (Space): Observation space definition
            action_space (Space): Action space definition
            precision (float): Action precision parameter
        """
        
    def update_beliefs(self, 
                      observation: np.ndarray) -> np.ndarray:
        """
        Update beliefs using [[knowledge_base/cognitive/predictive_processing|Predictive Processing]].
        
        Implementation: [[belief_updating]]
        
        Args:
            observation (np.ndarray): Current observation
            
        Returns:
            np.ndarray: Updated belief state
        """
        
    def select_action(self) -> np.ndarray:
        """
        Select action using Active Inference principles.
        
        Theory: [[knowledge_base/cognitive/active_inference#action-selection|Action Selection]]
        Implementation: [[action_selection]]
        
        Returns:
            np.ndarray: Selected action
        """
```

### Belief Updating System
See [[knowledge_base/cognitive/predictive_processing|Predictive Processing]] for theoretical foundation.

#### BeliefUpdater
```python
class BeliefUpdater:
    """
    Belief updating implementation.
    
    Theory: [[knowledge_base/cognitive/predictive_processing|Predictive Processing]]
    Examples: [[belief_updating_example]]
    """
    
    def __init__(self,
                 model: GenerativeModel,
                 inference_method: str = "variational"):
        """
        Initialize belief updater.
        
        Theory: [[knowledge_base/cognitive/free_energy_principle#variational-inference|Variational Inference]]
        
        Args:
            model (GenerativeModel): Generative model
            inference_method (str): Inference method
        """
        
    def update(self,
              prior: np.ndarray,
              likelihood: np.ndarray) -> np.ndarray:
        """
        Update beliefs using predictive processing.
        
        Theory: [[knowledge_base/cognitive/predictive_processing#belief-updating|Belief Updating]]
        Implementation: [[belief_updating]]
        
        Args:
            prior (np.ndarray): Prior beliefs
            likelihood (np.ndarray): Likelihood distribution
            
        Returns:
            np.ndarray: Posterior beliefs
        """
```

### Policy Selection
See [[action_selection]] for algorithm details.

#### PolicySelector
```python
class PolicySelector:
    """
    Policy selection implementation.
    
    See [[action_selection_example]] for usage.
    """
    
    def __init__(self,
                 policy_space: PolicySpace,
                 precision: float = 1.0):
        """
        Initialize policy selector.
        
        Args:
            policy_space (PolicySpace): Available policies
            precision (float): Selection precision
        """
        
    def select_policy(self,
                     beliefs: np.ndarray,
                     preferences: np.ndarray) -> Policy:
        """
        Select policy using expected free energy.
        
        Args:
            beliefs (np.ndarray): Current beliefs
            preferences (np.ndarray): Goal preferences
            
        Returns:
            Policy: Selected policy
        """
```

## Utility Functions

### Matrix Operations
```python
def compute_free_energy(beliefs: np.ndarray,
                       observations: np.ndarray) -> float:
    """
    Compute variational free energy.
    
    See [[free_energy_principle]] for theory.
    
    Args:
        beliefs (np.ndarray): Current beliefs
        observations (np.ndarray): Observed data
        
    Returns:
        float: Free energy value
    """

def compute_expected_free_energy(policy: Policy,
                               beliefs: np.ndarray) -> float:
    """
    Compute expected free energy for policy.
    
    See [[active_inference]] for details.
    
    Args:
        policy (Policy): Candidate policy
        beliefs (np.ndarray): Current beliefs
        
    Returns:
        float: Expected free energy
    """
```

### Visualization Tools
See [[visualization_tools]] for complete documentation.

```python
def plot_belief_state(beliefs: np.ndarray,
                     title: str = "Belief State") -> None:
    """
    Plot current belief state distribution.
    
    Args:
        beliefs (np.ndarray): Belief distribution
        title (str): Plot title
    """

def plot_action_selection(policies: List[Policy],
                         values: np.ndarray) -> None:
    """
    Plot policy selection process.
    
    Args:
        policies (List[Policy]): Available policies
        values (np.ndarray): Policy values
    """
```

## Data Structures

### State Spaces
```python
class BeliefState:
    """
    Belief state representation.
    
    See [[belief_updating]] for usage.
    """
    
    def __init__(self,
                 dimensions: Tuple[int, ...],
                 dtype: np.dtype = np.float32):
        """
        Initialize belief state.
        
        Args:
            dimensions (Tuple[int, ...]): State dimensions
            dtype (np.dtype): Data type
        """

class PolicySpace:
    """
    Policy space representation.
    
    See [[action_selection]] for usage.
    """
    
    def __init__(self,
                 action_space: Space,
                 horizon: int):
        """
        Initialize policy space.
        
        Args:
            action_space (Space): Action space
            horizon (int): Planning horizon
        """
```

## Integration Examples

### Basic Usage
```python
# Create agent
agent = ActiveInferenceAgent(
    observation_space=obs_space,
    action_space=action_space
)

# Update beliefs
observation = environment.observe()
beliefs = agent.update_beliefs(observation)

# Select action
action = agent.select_action()
```

See [[integration_examples]] for more examples.

## Error Handling

### Common Exceptions
```python
class DimensionError(Exception):
    """Raised when dimensions don't match."""
    pass

class ValidationError(Exception):
    """Raised when validation fails."""
    pass
```

See [[error_handling]] for details.

## Performance Considerations

### Optimization Guidelines
- Use vectorized operations
- Implement caching where appropriate
- Consider parallel processing for large models

See [[performance_optimization]] for details.

## Related Documentation
- [[api_documentation]]
- [[api_versioning]]
- [[api_examples]]
- [[package_documentation]]

## References
- [[active_inference]]
- [[free_energy_principle]]
- [[predictive_processing]]
- [[implementation_patterns]] 