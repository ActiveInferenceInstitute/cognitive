---
title: Cognitive Agents Index
type: agents
status: stable
created: 2026-02-06
updated: 2026-10-10
tags:
  - agents
  - index
  - cognitive_modeling
semantic_relations:
  - type: organizes
    links:
      - "[[code/Things/README]]"
      - "[[code/tools/src/models/active_inference/AGENTS]]"
      - "[[knowledge_base/AGENTS]]"
---

# Cognitive Agents Index

Entry point for the agent material in this repository. It separates what can
be imported and run from what is design documentation only. The authoritative
list of importable agents is [[docs/agents/agent_docs_readme|the agent documentation clearinghouse]].

## Agents that run

- **Discrete Active Inference runtime** (`code/tools/src/models/active_inference/`):
  `DiscreteGenerativeModel`, `ActiveInferenceDispatcher`, `HomeostaticFactory`,
  all importable from `cognitive`. [[code/tools/src/models/active_inference/AGENTS|Documentation]]
- **Simple POMDP** (`code/Things/Simple_POMDP/`): discrete Active Inference agent
  with YAML configuration. [[code/Things/Simple_POMDP/AGENTS|Documentation]]
- **Continuous Generic** (`code/Things/Continuous_Generic/`): continuous
  state-space agent, `Things.Continuous_Generic.ContinuousActiveInference`.
  [[code/Things/Continuous_Generic/AGENTS|Documentation]]

## Agent designs without code

These folders under `code/Things/` and `knowledge_base/` hold documentation
only. Treat them as design notes.

- **Generic POMDP** (`knowledge_base/agents/GenericPOMDP/`): configuration and
  state-space notes. [[knowledge_base/agents/GenericPOMDP/AGENTS|Documentation]]
- **Continuous time** (`knowledge_base/agents/Continuous_Time/`): continuous-time
  agent notes. [[knowledge_base/agents/AGENTS|Documentation]]
- **Ant Colony** (`code/Things/Ant_Colony/`): swarm and stigmergy design notes.
  [[code/Things/Ant_Colony/AGENTS|Documentation]]
- **BioFirm** (`knowledge_base/BioFirm/`): biological firm theory.
  [[knowledge_base/BioFirm/AGENTS|Documentation]]
- **Generic Thing** (`code/Things/Generic_Thing/`): base entity notes and saved images.
  [[code/Things/Generic_Thing/AGENTS|Documentation]]

The full folder index is [[code/Things/README]].

## Knowledge base

- [[knowledge_base/AGENTS|Active Inference Knowledge Base]]
- [[knowledge_base/cognitive/active_inference_agent|Active Inference Agent Theory]]
- [[knowledge_base/agents/architectures_overview|Agent Architectures Overview]]
- [[knowledge_base/agents/index|Agent Architecture Index]]

## Development tools

- [[code/tools/AGENTS|Cognitive tools instructions]]: the package source,
  console scripts, and test commands.

Most subdirectories of `code/Things/` and `knowledge_base/` have their own
`AGENTS.md` and `README.md`.
