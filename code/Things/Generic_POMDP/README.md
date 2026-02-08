# Generic POMDP Agent Framework

A flexible implementation of a Partially Observable Markov Decision Process (POMDP) agent using Active Inference.

## Contents

- **`AGENTS.md`**: Agent architecture and POMDP design details.
- **`Output/`**: Simulation results including matrix visualizations and EFE component plots.

For full source code and detailed documentation, see [[code/Things/Generic_POMDP/README|Knowledge Base: Generic POMDP]].

## Key Features

- **Discrete State Space**: Models the world as a set of discrete states.
- **Generative Model**: Uses A, B, C, D, E matrices to define the agent's world model.
- **Expected Free Energy**: Action selection via EFE minimization.

## Related Resources

- [[code/Things/Generic_POMDP/Generic_POMDP_README|Detailed Documentation]]
- [[knowledge_base/agents/GenericPOMDP/agent_config|Agent Configuration]]
- [[knowledge_base/agents/GenericPOMDP/matrices/A_matrix|A Matrix (Likelihood)]]
- [[knowledge_base/agents/GenericPOMDP/matrices/B_matrix|B Matrix (Transitions)]]
- [[knowledge_base/mathematics/active_inference_pomdp|Active Inference POMDP (Math)]]
- [[code/Things/Generic_POMDP/AGENTS|Agent Details]]
