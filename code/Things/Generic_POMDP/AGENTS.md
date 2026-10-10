---
title: Generic POMDP agent
type: agents
status: draft
updated: 2026-10-10
tags:
  - pomdp
  - discrete
semantic_relations:
  - type: documented_by
    links:
      - "[[code/Things/Generic_POMDP/README]]"
      - "[[code/Things/Generic_POMDP/Generic_POMDP_README]]"
  - type: related_to
    links:
      - "[[knowledge_base/mathematics/active_inference_pomdp]]"
---

# Generic POMDP agent

Documentation only; no runtime code lives in this folder and nothing can be
imported from it. The discrete loop it describes is implemented by
[[code/Things/Simple_POMDP/AGENTS|Simple_POMDP]].

The loop: perception updates beliefs about hidden states `s` from observations
`o` by Bayes' rule, and action selection prefers policies with lower expected
free energy `G`.

## Related

- [[knowledge_base/agents/GenericPOMDP/agent_config|Agent configuration]]
- [[knowledge_base/agents/GenericPOMDP/matrices/A_matrix|A matrix]]
- [[knowledge_base/agents/GenericPOMDP/matrices/B_matrix|B matrix]]
- [[knowledge_base/mathematics/expected_free_energy|Expected free energy]]
