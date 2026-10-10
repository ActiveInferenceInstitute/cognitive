---
title: Ant colony guide
type: guide
status: stable
updated: 2026-10-10
tags:
  - ant_colony
  - swarm_intelligence
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[code/Things/Ant_Colony/AGENTS]]"
      - "[[knowledge_base/cognitive/swarm_intelligence]]"
      - "[[knowledge_base/biology/README]]"
---

# Ant colony guide

## What exists

`code/Things/Ant_Colony/` holds documentation only (`README.md`, `AGENTS.md`,
`Ant_Colony_README.md`). There is no `ant_colony` package, no `main.py`, and no
`colony_config.yaml` in this repository, so this guide gives no run commands
and no configuration table. The importable agents are `SimplePOMDP`,
`ContinuousActiveInference`, and `HomeostaticInference`; see
[[docs/api/agent_api|Agent API reference]].

## Where the material lives

- Research write-up: [[docs/research/ant_colony_active_inference|Ant colony and Active Inference]].
- Biology: [[knowledge_base/biology/ant_colony_organization|Ant colony organization]],
  [[knowledge_base/biology/ant_communication_systems|Ant communication systems]],
  [[knowledge_base/biology/foraging_optimization|Foraging optimization]].
- Theory: [[knowledge_base/cognitive/stigmergic_coordination|Stigmergic coordination]],
  [[knowledge_base/cognitive/swarm_intelligence|Swarm intelligence]],
  [[knowledge_base/cognitive/collective_behavior|Collective behavior]].
- Study sequence: [[docs/guides/learning_paths/active_inference_myrmecology_learning_path|Myrmecology learning path]].
- Working code to start from: [[docs/examples/ant_colony_example|Ant colony usage example]],
  which points to the runnable examples.

## Building a colony model on this package

A colony model would sit outside this repository and depend on it. The
building block is one `DiscreteGenerativeModel` per ant, with observations,
states, and actions defined by the modeller (for example local cue levels,
position relative to nest and food, and moves). Every environment, pheromone
field, and multi-agent loop in such a project is the modeller's own code.
