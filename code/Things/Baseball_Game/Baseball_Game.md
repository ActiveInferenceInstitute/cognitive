---
title: Baseball Game Concept Notes
type: design_notes
status: archived
created: 2025-01-01
updated: 2026-10-10
tags:
  - baseball
  - multi_agent
  - game_theory
  - design
semantic_relations:
  - type: relates
    links:
      - "[[code/Things/Baseball_Game/README|Baseball Game readme]]"
      - "[[knowledge_base/cognitive/decision_making|Decision making]]"
      - "[[knowledge_base/mathematics/game_theory|Game theory]]"
      - "[[knowledge_base/cognitive/multi_agent_active_inference|Multi-agent active inference]]"
---

# Baseball Game: Concept Notes

> **Status: archived concept study; no code in this repository.** `code/Things/Baseball_Game/` holds documentation only (`README.md`, `AGENTS.md`, this page). There is no simulator, physics engine, agent class, visualization or test. The material below is a design sketch of how baseball could be framed as a multi-agent Active Inference problem.

## Framing

Baseball is a useful thought experiment for Active Inference because it combines physical dynamics (ball flight, player motion), discrete game state (inning, count, outs, base occupancy) and strategic interaction between two teams under uncertainty.

## Agents

- **Players:** each an Active Inference agent with position-specific priors and policies, observing through proprioceptive and exteroceptive channels.
- **Teams:** a higher-level agent whose policies (defensive positioning, batting order, pitching changes) constrain the players below it.
- **Umpires:** observers who enforce rules and so shape which outcomes are admissible.

## State space

- **Physical:** ball position, velocity and spin; player positions; field geometry; weather.
- **Game:** inning, score, balls and strikes, outs, base occupancy.
- **Mental (latent):** confidence, momentum, strategic intent, risk assessment.

## Hierarchical generative model

1. **Low level:** ball trajectories, collisions and player movement.
2. **Mid level:** play outcomes and situation-specific strategies.
3. **High level:** game flow, win probability and long-term planning.

Higher levels would supply priors to lower levels, with action selection by expected free energy at each level.

## Questions the framing could address

- Do team-level strategies emerge from belief alignment between players?
- How do precision (confidence) changes alter risk taking in late innings?
- How would skill acquisition appear as learning of the likelihood and transition model?

## Related

- Executable agents: `code/Things/Simple_POMDP/` and `code/Things/Continuous_Generic/` (see [[docs/api/api_reference|API reference]]).
- Theory: [[knowledge_base/cognitive/decision_making|decision making]], [[knowledge_base/mathematics/game_theory|game theory]].
