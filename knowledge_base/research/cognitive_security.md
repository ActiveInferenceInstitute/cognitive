---
title: Cognitive Security
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - research
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/research/infohazard_management]]"
      - "[[knowledge_base/cognitive/precision_weighting]]"
      - "[[knowledge_base/cognitive/multi_agent_active_inference]]"
      - "[[knowledge_base/research/security_evolution]]"
---

# Cognitive Security

## Overview

Cognitive security concerns the integrity of the processes by which people and groups form beliefs and make decisions, and how those processes can be degraded or exploited. It overlaps with information security but targets the receiver's inference rather than the system that stores data. The term is used by practitioner communities and in defence literature, and it is not a settled academic discipline with a single definition.

## Main questions

- **Vocabulary of harm.** Wardle and Derakhshan's *Information Disorder* (2017) separates misinformation (false, no intent to harm), disinformation (false, intended to harm) and malinformation (true, used to harm).
- **Defence.** Fact-checking, source labelling, and "prebunking", which draws on McGuire's inoculation theory: exposing people to weakened forms of a manipulation technique to build resistance. Evidence for prebunking is encouraging in controlled studies, but durability and real-world effect sizes are still being measured.
- **Ecosystem structure.** How network topology, platform design and incentives shape the spread of beliefs, including echo chambers and polarisation.
- **Trust.** Who is believed, how reliability is estimated, and how it is attacked.

## Connection to Active Inference

The link is a modelling programme with some published formal work, not a validated account of manipulation. In Active Inference, beliefs are posterior distributions, and the weight given to a source or a prediction error is its estimated precision. This gives a compact description of several attack surfaces: an adversary can supply misleading evidence, can erode the perceived reliability of correct sources, or can inflate confidence in unreliable ones. Trust in another agent's testimony is a precision on that testimony.

At the group level, agents that communicate and share generative models can converge on common beliefs, which makes coordination possible and also creates the conditions for echo chambers. Collective-inference models in the Active Inference literature, such as Kaufmann and colleagues' (2021) model of collective intelligence and Heins and colleagues' (2022) spin-glass formulation, treat groups as coupled inferential systems. These are theoretical models with simulated demonstrations. Their predictions about real misinformation dynamics have not been established empirically, and conventional epidemic-style and agent-based models of information spread remain the main quantitative tools.

## In this repository

Related pages cover adversarial and hazardous information ([[knowledge_base/research/infohazard_management]]), the evolution of defensive mechanisms ([[knowledge_base/research/security_evolution]]), and the machinery used to model reliability ([[knowledge_base/cognitive/precision_weighting]]). Group-level modelling is in [[knowledge_base/cognitive/multi_agent_active_inference]] and [[knowledge_base/research/social_cognition]]. Study material is in [[active_inference_security_cognitive_systems_path]].

## References

- Wardle, C., & Derakhshan, H. (2017). *Information Disorder: Toward an Interdisciplinary Framework for Research and Policy Making*. Council of Europe.
- Kaufmann, R., Gupta, P., & Taylor, J. (2021). An active inference model of collective intelligence. *Entropy*, 23(7), 830.
- Heins, C., et al. (2022). Spin glass systems as collective active inference. arXiv.

## Related Resources

- [[knowledge_base/cognitive/active_inference]]
- [[knowledge_base/research/infohazard_management]]
