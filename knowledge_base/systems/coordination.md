---
title: Coordination
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - coordination
  - multi_agent
  - synchronization
  - joint_action
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/cognitive/stigmergic_coordination]]"
      - "[[knowledge_base/cognitive/sensorimotor_coordination]]"
      - "[[knowledge_base/cognitive/cooperation]]"
      - "[[knowledge_base/mathematics/game_theory]]"
---

# Coordination

## Overview
Coordination is the alignment of the actions of two or more agents, or of several components of one agent, so that their joint effect is appropriate to a shared goal or constraint. It differs from cooperation: coordination concerns compatibility of actions, and agents with conflicting interests can still face a coordination problem.

## Mechanism
Mechanisms can be grouped by what carries the information. Explicit communication transmits intentions directly. Conventions resolve coordination games with several equilibria, such as which side of the road to drive on (Lewis 1969), and focal points let agents converge without communication when some option stands out (Schelling 1960). Hierarchy and leadership assign roles from above. Stigmergy coordinates through traces left in the environment, as with ant trails. Synchronization, in which oscillators entrain through coupling, is modeled by the Kuramoto model: d(theta_i)/dt = omega_i + (K/N) sum_j sin(theta_j - theta_i), which shows a transition to collective synchrony when the coupling K exceeds a critical value; firefly flashing and some neural rhythms are discussed in these terms. Consensus algorithms in distributed computing reach agreement using local exchanges.

Each approach has costs. Direct communication uses bandwidth and fails when channels are lost. Convention requires prior shared history. Centralized control is a single point of failure. Environmental mediation is slow but scales to large groups with minimal individual capacity. Within one body, motor coordination solves a similar problem across many degrees of freedom, treated in [[knowledge_base/cognitive/sensorimotor_coordination]].

## Role in Active Inference
Active Inference gives a particular account in which coordination follows from agents that predict one another. Two agents each equipped with a generative model that includes the other can reach synchrony because each acts to make its sensations, which include the partner's behavior, match its predictions; Friston and Frith (2015) described this as generalized synchrony between coupled systems in a paper on a duet for one. The repository's multi-agent page extends this to sets of agents. The approach is demonstrated in simulation and in a few empirical studies of interpersonal synchrony, and its explanatory edge over standard coupled-oscillator or game-theoretic accounts is still being assessed.

## Related concepts
- [[knowledge_base/cognitive/stigmergic_coordination]] — coordination via the environment
- [[knowledge_base/cognitive/multi_agent_active_inference]] — formal multi-agent treatment
- [[knowledge_base/cognitive/neural_synchrony]] — synchronization within the brain
- [[knowledge_base/cognitive/cooperation]] — shared-goal behavior built on coordination
- [[knowledge_base/systems/social_coordination]] — coordination through norms, roles, and institutions
- [[knowledge_base/systems/task_allocation]] — coordination of who does what

## References
- Lewis, D. (1969). Convention: A Philosophical Study.
- Schelling, T. C. (1960). The Strategy of Conflict.
- Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence.
- Friston, K., and Frith, C. (2015). A duet for one. Consciousness and Cognition.
