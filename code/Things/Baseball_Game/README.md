---
title: Baseball Game
type: design_notes
status: archived
created: 2025-01-01
updated: 2026-10-10
tags:
  - baseball
  - game
  - decision_making
  - design
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/cognitive/decision_making|Decision making]]"
      - "[[docs/guides/application/README|Application guides]]"
      - "[[code/Things/Baseball_Game/Baseball_Game|Baseball Game concept notes]]"
---

# Baseball Game

> **Status: archived concept study; no code in this repository.** This folder contains documentation only: this page, `AGENTS.md` and `Baseball_Game.md`. There is no `BaseballEnvironment`, `BaseballRulesEngine`, cognitive baseball agent, analytics system, physics simulation or runnable example. Earlier versions of this page showed Python for those classes; none of it existed and it has been removed.

## What the folder is

A written exploration of how a baseball game, with physical dynamics, discrete game state and strategic two-team interaction, could be framed as a multi-agent Active Inference problem. The framing is in [[code/Things/Baseball_Game/Baseball_Game|Baseball Game concept notes]].

## If you want to run something

- Discrete-state agent with tests: `code/Things/Simple_POMDP/`.
- Continuous-state agent: `code/Things/Continuous_Generic/`.
- Package reference: [[docs/api/api_reference|API reference]]; overview of the Things: [[docs/agents/agent_docs_readme|agent documentation]].

## Related theory

- [[knowledge_base/cognitive/decision_making|Decision making]]
- [[knowledge_base/mathematics/game_theory|Game theory]]
- [[code/Things/Generic_Thing/generic_thing|Generic Thing design notes]]
