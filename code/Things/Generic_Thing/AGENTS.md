---
title: Generic Thing
type: agents
status: core
tags:
  - base_class
  - entity
  - active_inference
semantic_relations:
  - type: documented_by
    links:
      - [[code/Things/Generic_Thing/README]]
      - [[code/Things/Generic_Thing/generic_thing]]
---

# Generic Thing

The fundamental unit of the framework. See [[code/Things/Generic_Thing/README|Knowledge Base]] for full implementation details.

## Definition

A "Thing" in this context is any entity that can:

1. Perceive its environment (receive observations).
2. Act upon its environment (emit actions).
3. Maintain internal states (beliefs).

## Related

- [[knowledge_base/cognitive/active_inference_agent|Active Inference Agent Theory]]
- [[knowledge_base/cognitive/generative_model|Generative Model]]
- [[knowledge_base/cognitive/belief_updating|Belief Updating]]
