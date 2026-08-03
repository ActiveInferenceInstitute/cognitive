---
title: Agent Architectures Documentation
type: agents
status: stable
created: 2024-02-07
updated: 2026-08-02
tags:
  - agents
  - architectures
  - active_inference
  - cognitive_agents
  - implementation
semantic_relations:
  - type: organizes
    links:
      - '[[README|Agent architectures overview]]'
      - '[[architectures_overview|Architectures overview]]'
      - '[[index|Agent index]]'
      - '[[GenericPOMDP/README|Generic POMDP]]'
      - '[[Continuous_Time/README|Continuous time]]'
      - '[[../../code/Things/AGENTS|Things index]]'
---

# Agent Architectures Documentation

Conceptual documentation for agent architectures in the framework: POMDP-based
agents, continuous-time agents, and the design patterns behind them.

## Documents

- [[architectures_overview]] — cross-cutting architecture patterns.
- [[index]] — agent architecture index.
- `GenericPOMDP/` — the flexible POMDP framework (notes under
  `GenericPOMDP/AGENTS.md`).
- `Continuous_Time/` — continuous state-space agents (notes under
  `Continuous_Time/AGENTS.md`).

## Notes

- Executable agent implementations live in `code/Things/`; see
  `code/Things/AGENTS.md` for the index.
- The validated discrete model family is in
  `code/tools/src/models/active_inference/`.
- These pages are conceptual; pseudocode within them is illustrative and not
  package API.
