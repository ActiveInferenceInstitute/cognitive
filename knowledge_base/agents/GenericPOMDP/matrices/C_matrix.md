---
type: matrix_spec
id: C_matrix_001
matrix_type: preference
created: {{date}}
modified: {{date}}
tags: [matrix, preference, active-inference]
related_spaces: [o_space]
---

# C-Matrix (Preferences/Goals)

## Definition
The C-matrix defines the agent's preferences over observations, representing expected or desired observations at different time points in the future.

## Matrix Structure
```yaml
dimensions:
  rows: num_observations    # From [[o_space]]
  cols: num_time_points    # Planning horizon
shape_constraints:
  - rows > 0
  - cols > 0
  - finite_values  # Typically log-probabilities
```

## Mathematical Form
```python
# Preference encoding
C[o,τ] = ln P(o_τ)  # Log-probability of desired observation o at time τ

# Properties
C ∈ ℝ              # Real-valued preferences
exp(C) ∈ [0,1]     # Corresponding probabilities
```

## Data Structure
```yaml
matrix_data:
  format: numpy.ndarray
  dtype: float32
  shape: [num_observations, num_time_points]
  initialization: neutral  # Start with uniform preferences
  storage: matrix_store/C_matrix.npy
```

## Visualization
```yaml
plot_type: heatmap
colormap: "RdYlGn"  # Red (negative) to Green (positive)
title: "Preference Matrix (C)"
xlabel: "Time Steps"
ylabel: "Observations"
```

## Properties
### Temporal Structure
- Preferences may change over time
- [[temporal_discounting]]
- [[goal_hierarchy]]

### Value Encoding
- Log-probability format
- Relative preferences
- [[utility_theory]]

### Optimization Targets
- Drive policy selection
- Guide exploration
- [[reward_function]]

## Update Rules
- [[preference_learning]] - Learning from experience
- [[goal_adaptation]] - Dynamic preference updating
- [[context_modulation]] - Context-dependent preferences

## Integration
- Guides [[policy_selection]]
- Affects [[expected_free_energy]]
- Defines [[value_function]]

## Examples
```python
# Example preference matrix for 3 observations over 4 time steps
C = np.array([
    [-1.0, -1.0, -1.0, -1.0],  # Avoid observation 1
    [ 0.0,  0.0,  0.0,  0.0],  # Neutral about observation 2
    [ 1.0,  1.0,  1.0,  1.0]   # Prefer observation 3
])

# Time-varying preferences
C_dynamic = np.array([
    [ 0.0, -1.0, -2.0, -3.0],  # Increasingly avoid
    [ 0.0,  0.0,  0.0,  0.0],  # Remain neutral
    [ 0.0,  1.0,  2.0,  3.0]   # Increasingly prefer
])
```

## Computational Interface
```python
class PreferenceMatrix:
    def __init__(self, num_observations: int, time_horizon: int):
        self.C = self._initialize_preferences(num_observations, time_horizon)
    
    def get_preferences(self, time_step: int) -> np.ndarray:
        """Get preferences for given time step"""
        return self.C[:, time_step]
    
    def update_preferences(self,
                         observation: int,
                         time_step: int,
                         value: float):
        """Update preference value"""
        pass
    
    def compute_expected_value(self, 
                             belief: np.ndarray,
                             time_step: int) -> float:
        """Compute expected value given belief state"""
        pass
```

## Learning Methods
- [[inverse_reinforcement]] - Learning from behavior
- [[preference_inference]] - Inferring preferences
- [[value_learning]] - Learning values from experience

## Applications
### Goal-Directed Behavior
- [[goal_setting]]
- [[motivation_modeling]]
- [[reward_shaping]]

### Planning
- [[trajectory_optimization]]
- [[hierarchical_planning]]
- [[risk_sensitive_control]]

## Related Components
- [[o_space]] - Observation space
- [[value_function]] - Value computation
- [[policy_selection]] - Action selection
- [[goal_specification]] - Goal definition

## References
- [[utility_theory]]
- [[active_inference_control]]
- [[preference_learning]] 