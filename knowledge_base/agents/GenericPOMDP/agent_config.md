---
type: agent_config
id: generic_pomdp_001
created: {{date}}
modified: {{date}}
tags: [agent, pomdp, active-inference]
---

# Generic POMDP Agent

## Model Structure
This agent implements a POMDP (Partially Observable Markov Decision Process) using Active Inference principles.

### Generative Model Components
- [[A_matrix]] - Likelihood/Perception matrix (observation mapping)
- [[B_matrix]] - Transition/Dynamics matrix (state transitions)
- [[C_matrix]] - Preference/Cost matrix (goal states)
- [[D_matrix]] - Prior beliefs matrix (initial states)
- [[E_matrix]] - Affordance/Policy matrix (allowable actions)

### State Spaces
- [[o_space]] - Observation space
- [[s_space]] - Hidden state space
- [[pi_space]] - Policy space

## Matrix Dimensions
```yaml
dimensions:
  A: [num_observations, num_states]
  B: [num_states, num_states, num_actions]
  C: [num_observations, num_time_points]
  D: [num_states]
  E: [num_policies, num_actions]
```

## Implementation Details
- [[generative_model]] - Full model specification
- [[inference_scheme]] - Belief updating mechanism
- [[policy_selection]] - Action selection method

## Visualization
- [[matrix_plots]] - Matrix visualizations
- [[state_evolution]] - State trajectory plots
- [[free_energy_landscape]] - Free energy surface plots

## Integration
- [[simulation_config]] - Simulation parameters
- [[environment_interface]] - Environment coupling
- [[logging_config]] - Data logging setup

## References
- [[active_inference_theory]]
- [[pomdp_formalism]]
- [[matrix_operations]] 