---
title: Technology Ethics
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - systems
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/systems/digital_governance]]"
      - "[[knowledge_base/philosophy/dark_room_problem]]"
      - "[[knowledge_base/free_energy_principle/applications/ai_safety]]"
---

# Technology Ethics

## Overview

Technology ethics asks what obligations designers, deployers and institutions have toward the people affected by a system. For autonomous agents the recurring themes are transparency, accountability, autonomy and value alignment. Active Inference does not resolve these questions, and nothing in the framework makes an agent ethical. It does change where some of the problems appear, because an agent's goals and beliefs are explicit objects. The points below are design considerations, not findings from field studies.

## Themes

**Transparency.** In a discrete Active Inference agent the preference vector $C$, the priors, the likelihood and the current beliefs can all be read out and logged, so one can ask what the agent believes and wants. This helps auditing. It stops short of explanation, however: an agent with a very large learned model remains hard to interpret, and knowing the preferences does not tell an affected person why a decision was made in their case.

**Accountability.** A responsible party must be identifiable for each goal specification. Because preferences are set by designers or learned from data, accountability attaches to those choices and to the decision to deploy. Logging of beliefs, policies and overrides makes retrospective review possible ([[knowledge_base/systems/digital_governance]]).

**Autonomy.** Two directions matter: the autonomy of the people the system affects, and the autonomy granted to the agent. An agent that infers a user's preferences and then steers toward them can narrow the user's options. A system that treats the user as a source of information to be made predictable has an incentive that conflicts with the user's interests; since minimising expected surprise can be satisfied by changing the world or the person, designers should state what the agent is permitted to change.

**Value alignment.** Misspecified preferences are pursued competently. The "dark room" problem, the worry that surprise minimisation favours trivial, unchanging environments, is addressed in the framework by preferences and epistemic value, but the underlying lesson holds: specify what the agent should seek, and test what it does when the specification is incomplete ([[knowledge_base/philosophy/dark_room_problem]]). Learning preferences from human behaviour is under-determined and can encode existing biases.

**Curiosity and privacy.** Epistemic drive pushes an agent to reduce uncertainty. Applied to people, this becomes data collection, so information-seeking needs explicit limits and consent.

## Practices

Impact assessment before deployment, value-sensitive design that consults affected stakeholders (Friedman and colleagues), adversarial testing, human override for high-stakes actions, and monitoring after release.

## In This Repository

Related pages: [[knowledge_base/systems/digital_governance]], [[knowledge_base/systems/risk_assessment]], [[knowledge_base/research/cognitive_security]], [[knowledge_base/research/infohazard_management]], [[knowledge_base/research/superintelligence]] and [[knowledge_base/free_energy_principle/applications/ai_safety]]. The code is educational: agents act in small simulated environments with no real-world effects. It offers inspectable state (`ModelState`, saved with `save_state`) and validated inputs, which are the raw material for transparency, but no consent, privacy or oversight mechanism.

## Related Resources

- [[knowledge_base/cognitive/active_inference]]
- [[knowledge_base/systems/digital_governance]]
