---
title: Ant Colony Design Notes
type: design_notes
status: draft
created: 2025-01-01
updated: 2026-10-10
tags:
  - ant_colony
  - multi_agent
  - stigmergy
  - design
semantic_relations:
  - type: relates
    links:
      - "[[code/Things/Ant_Colony/README|Ant Colony folder]]"
      - "[[knowledge_base/biology/ant_colony_organization|Ant colony organization]]"
      - "[[knowledge_base/cognitive/swarm_intelligence|Swarm intelligence]]"
      - "[[docs/agents/agent_docs_readme|Agent documentation]]"
---

# Ant Colony: Design Notes

> **Status: design sketch; no code in this repository.** The `code/Things/Ant_Colony/` folder contains documentation only (`README.md`, `AGENTS.md`, this page). There is no `ant_colony` package, no `Nestmate`, `Colony`, `Environment` or `Simulation` class, no YAML configuration and no tests. The only executable Things are `Simple_POMDP` and `Continuous_Generic` (see [[docs/api/api_reference|API reference]]).

## Idea

Model an ant colony as a multi-agent system in which each ant (a "nestmate") is an Active Inference agent. Colony-level behaviour (trail formation, task allocation) is expected to emerge from individual free energy minimization coupled through a shared environment, rather than from central control. This is a conceptual proposal, not a result obtained in this repository.

## Proposed components

- **Nestmate agent.** Observes local pheromone concentration, food, obstacles and neighbours; holds a generative model of its surroundings; selects actions by minimizing expected free energy; deposits pheromone as an action.
- **Colony environment.** A shared grid with pheromone deposition, diffusion and evaporation, food sources and obstacles. Pheromone acts as the stigmergic channel: agents communicate only by changing the environment.
- **Multi-agent layer.** Spatial interaction between nestmates and task allocation (foraging, nest maintenance, brood care, defence) arising from differences in beliefs and preferences.

## Open modelling questions

- How should pheromone fields enter the observation likelihood (A matrix): as a graded discrete observation or a continuous one?
- Which preferences (C) produce trail following versus exploration, and how are they learned?
- How does task switching arise: from precision changes, from different priors, or from explicit role states?
- Which collective measures (task distribution, interaction networks, synchrony) would show emergence rather than assume it?

## Where to start in this repository

- Discrete-state agents: `code/Things/Simple_POMDP/` (executable, with tests).
- Continuous-state agents: `code/Things/Continuous_Generic/` (executable).
- Theory: [[knowledge_base/biology/ant_colony_organization|ant colony organization]], [[knowledge_base/biology/ant_communication_systems|ant communication systems]], [[knowledge_base/cognitive/swarm_intelligence|swarm intelligence]], [[knowledge_base/cognitive/collective_behavior|collective behavior]].
