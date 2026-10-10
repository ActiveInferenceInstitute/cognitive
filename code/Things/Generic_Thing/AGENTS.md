---
title: Generic Thing
type: agents
status: draft
updated: 2026-10-10
tags:
  - entity
  - active_inference
semantic_relations:
  - type: documented_by
    links:
      - "[[code/Things/Generic_Thing/README]]"
      - "[[code/Things/Generic_Thing/generic_thing]]"
---

# Generic Thing

Documentation only; nothing can be imported from this folder.

A "Thing" in this vocabulary is any entity that can:

1. Perceive its environment (receive observations).
2. Act upon its environment (emit actions).
3. Maintain internal states (beliefs).

The executable agents are [[code/Things/Simple_POMDP/AGENTS|Simple_POMDP]] and
[[code/Things/Continuous_Generic/AGENTS|Continuous_Generic]].

## Related

- [[knowledge_base/cognitive/active_inference_agent|Active inference agent theory]]
- [[knowledge_base/cognitive/generative_model|Generative model]]
- [[knowledge_base/cognitive/belief_updating|Belief updating]]
