---
title: Generic POMDP
type: concept
status: draft
updated: 2026-10-10
semantic_relations:
  - type: documented_by
    links:
      - "[[code/Things/Generic_POMDP/Generic_POMDP_README]]"
      - "[[code/Things/Generic_POMDP/AGENTS]]"
  - type: related_to
    links:
      - "[[knowledge_base/agents/GenericPOMDP/agent_config|Agent configuration]]"
      - "[[knowledge_base/mathematics/active_inference_pomdp|Active inference POMDP]]"
---

# Generic POMDP

Design sketch; no code in this repository. This folder contains documentation
only: `README.md`, `AGENTS.md`, and `Generic_POMDP_README.md`. There is no
`Output/` directory and no importable module.

The page sketches a discrete agent configured by A, B, C, D, E matrices that
selects actions by minimizing expected free energy. The working implementation
of that structure is [[code/Things/Simple_POMDP/README|Simple_POMDP]] and, at the
package level, `DiscreteGenerativeModel` and `ActiveInferenceDispatcher` (see
[[docs/api/api_reference|the API reference]]).

## Related

- [[code/Things/Generic_POMDP/Generic_POMDP_README|Design notes]]
- [[knowledge_base/agents/GenericPOMDP/agent_config|Agent configuration]]
- [[knowledge_base/agents/GenericPOMDP/matrices/A_matrix|A matrix (likelihood)]]
- [[knowledge_base/agents/GenericPOMDP/matrices/B_matrix|B matrix (transitions)]]
