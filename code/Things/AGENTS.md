---
title: Things index
type: agents
status: stable
updated: 2026-10-10
tags:
  - index
  - agents
  - active_inference
semantic_relations:
  - type: contains
    links:
      - "[[code/Things/Simple_POMDP/AGENTS]]"
      - "[[code/Things/Continuous_Generic/AGENTS]]"
      - "[[code/Things/Ant_Colony/AGENTS]]"
      - "[[code/Things/BioFirm/AGENTS]]"
      - "[[code/Things/Generic_POMDP/AGENTS]]"
      - "[[code/Things/Generic_Thing/AGENTS]]"
      - "[[code/Things/Baseball_Game/AGENTS]]"
      - "[[code/Things/KG_Multi_Agent/AGENTS]]"
      - "[[code/Things/Path_Network/AGENTS]]"
      - "[[code/Things/ActiveInferenceInstitute/AGENTS]]"
---

# Things index

`code/Things/` is the `Things` package. Two folders hold executable agents; the
other eight hold documentation only. For architecture background see
[[knowledge_base/agents/architectures_overview|Architectures overview]]; for the
exact signatures see [[docs/api/api_reference|the API reference]].

## Executable agents

- [[code/Things/Simple_POMDP/AGENTS|Simple POMDP]]: `from Things.Simple_POMDP import SimplePOMDP`.
- [[code/Things/Continuous_Generic/AGENTS|Continuous Generic]]: `from Things.Continuous_Generic import ContinuousActiveInference, ContinuousVisualizer`.

## Documentation-only folders (no importable code)

- [[code/Things/Ant_Colony/AGENTS|Ant Colony]]
- [[code/Things/BioFirm/AGENTS|BioFirm]]
- [[code/Things/Generic_POMDP/AGENTS|Generic POMDP]]
- [[code/Things/Generic_Thing/AGENTS|Generic Thing]] (also holds saved PNG images)
- [[code/Things/Baseball_Game/AGENTS|Baseball Game]]
- [[code/Things/KG_Multi_Agent/AGENTS|KG Multi-Agent]]
- [[code/Things/Path_Network/AGENTS|Path Network]]
- [[code/Things/ActiveInferenceInstitute/AGENTS|Active Inference Institute]]

## Rules for this folder

A page here names a class, script, or configuration file only if it exists in
the repository. Documentation-only folders say so near the top.
