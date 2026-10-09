---
title: Agent documentation clearinghouse
type: documentation
status: stable
updated: 2026-10-09
semantic_relations:
  - type: organizes
    links:
      - "[[docs/api/agent_api|Agent API]]"
      - "[[docs/api/api_reference|API reference]]"
      - "[[code/Things/AGENTS|Things index]]"
      - "[[knowledge_base/agents/index|Agent knowledge base]]"
---

# Agent documentation clearinghouse

A navigation page. It states which agents can be imported and run, and points to
the conceptual material for the rest. Every entry below exists in the
repository.

## Agents that run

| Agent | Import | Source | Documentation |
|---|---|---|---|
| Discrete model and dispatcher | `from cognitive import DiscreteGenerativeModel, ActiveInferenceDispatcher, InferenceConfig` | `code/tools/src/models/active_inference/` | [[docs/api/api_reference]] |
| Homeostatic agent | `from cognitive import HomeostaticFactory` | `code/tools/src/models/active_inference/homeostatic.py` | [[docs/api/agent_api]] |
| Simple POMDP | `from Things.Simple_POMDP import SimplePOMDP` | `code/Things/Simple_POMDP/` | [[code/Things/Simple_POMDP/AGENTS]] |
| Continuous agent | `from Things.Continuous_Generic import ContinuousActiveInference` | `code/Things/Continuous_Generic/` | [[code/Things/Continuous_Generic/AGENTS]] |

Runnable examples for each are in [[docs/api/api_examples|API examples]].

## Agent folders without code

These folders under `code/Things/` hold documentation (and, for `Generic_Thing`,
saved images) but no importable Python: `Ant_Colony`, `BioFirm`,
`Generic_POMDP`, `Generic_Thing`, `Baseball_Game`, `KG_Multi_Agent`,
`Path_Network`, `ActiveInferenceInstitute`. Treat their pages as design notes,
not as descriptions of an installed API. The index is [[code/Things/AGENTS|Things index]].

## Conceptual material

- [[knowledge_base/agents/index|Agent knowledge base index]] and
  [[knowledge_base/agents/architectures_overview|architectures overview]].
- [[docs/guides/agent_development|Agent development guide]] and
  [[docs/examples/agent_examples|agent examples]].
- [[docs/README|Documentation hub]] for the rest of `docs/`.

## Rules for this folder

See [[AGENTS|agent documentation instructions]]. A page here names a class only
if it can be imported, and gives the import path.
