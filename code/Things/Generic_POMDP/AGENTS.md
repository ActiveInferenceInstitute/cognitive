---
title: Generic POMDP Agent
type: agents
status: stable
tags:
  - pomdp
  - discrete
  - active_inference
semantic_relations:
  - type: implements
    links:
      - [[knowledge_base/mathematics/active_inference_pomdp]]
---

# Generic POMDP Agent

This agent implements the standard discrete-time Active Inference loop. See [[code/Things/Generic_POMDP/README|Knowledge Base]] for full implementation details.

## Architecture

- **Perception**: Updates beliefs about hidden states ($s$) based on observations ($o$) roughly via $P(s|o) \propto P(o|s)P(s)$.
- **Action**: Selects policies ($\pi$) that minimize Expected Free Energy ($G$).

## Related

- [[knowledge_base/agents/GenericPOMDP/agent_config|Agent Configuration]]
- [[knowledge_base/agents/GenericPOMDP/matrices/A_matrix|A Matrix]]
- [[knowledge_base/agents/GenericPOMDP/matrices/B_matrix|B Matrix]]
- [[knowledge_base/mathematics/expected_free_energy|Expected Free Energy]]
