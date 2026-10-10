---
title: Multi-agent architecture
type: research
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/cognitive/multi_agent_active_inference]]"
      - "[[knowledge_base/cognitive/swarm_intelligence]]"
      - "[[docs/api/ant_colony_api]]"
---

# Multi-agent architecture

Not built. There is no environment, communication channel, or population
class in the package. The folders `code/Things/Ant_Colony/`,
`code/Things/KG_Multi_Agent/`, and `code/Things/Path_Network/` contain
documentation (and, for `KG_Multi_Agent`, a documentation subfolder) but no
importable Python; see [[docs/api/ant_colony_api|the ant colony API page]].

A population can be built today by creating several `SimplePOMDP` or
`ContinuousActiveInference` instances and stepping them in a loop of your own.
The theory is in [[knowledge_base/cognitive/multi_agent_active_inference]] and
[[knowledge_base/cognitive/swarm_intelligence]]; a design proposal for the ant
colony case is in [[docs/research/ant_colony_active_inference|the ant colony proposal]].
