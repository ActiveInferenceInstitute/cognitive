---
title: AGI Development
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
      - "[[knowledge_base/cognitive/predictive_processing]]"
      - "[[knowledge_base/research/superintelligence]]"
      - "[[knowledge_base/free_energy_principle/applications/ai_safety]]"
      - "[[knowledge_base/cognitive/hierarchical_models]]"
aliases:
  - artificial_general_intelligence
wikidata: Q2264109
---

# AGI Development

## Overview

Artificial general intelligence (AGI) has no agreed definition. In common usage it means a system that can learn and perform a wide range of cognitive tasks at roughly human level or better, as opposed to a system built for one task. Whether any existing system qualifies, and how one would test for it, is disputed. This page therefore treats AGI as a research aim and asks one narrower question: what does Active Inference contribute to the design of general, adaptive agents?

## Main approaches

Current work falls into several families that overlap in practice.

- **Scaling of learned models.** Large neural networks trained on broad data, sometimes with reinforcement learning or tool use added on top.
- **Reward-based agents.** Reinforcement learning, with the AIXI agent of Marcus Hutter (2005) as an idealised, incomputable reference point for "optimal" general agency.
- **Cognitive architectures.** Hand-designed systems such as Soar and ACT-R that combine memory, goal management and learning modules.
- **Neuroscience-inspired designs.** Models that borrow organising principles from the brain, such as hierarchy, prediction and active sensing.

## Connection to Active Inference

The connection is an active research programme, not an established route to AGI. Active Inference specifies an agent as a generative model together with a single objective, variational free energy for perception and learning and expected free energy for action selection. Expected free energy combines a pragmatic term (reaching preferred outcomes) and an epistemic term (resolving uncertainty), so exploration arises from the same objective as goal pursuit rather than from an added bonus. This differs from standard reinforcement learning, where reward is the primary quantity and exploration is typically handled separately.

Three features make the framework relevant to general agents. Hierarchical and temporally deep generative models offer a route to planning at several time scales. Preferences are expressed as prior beliefs about outcomes, which is a different parameterisation of goals than a scalar reward. Action and perception are treated as one inferential process, which suits embodied agents.

The open problem is scale. Published implementations are mostly small discrete state-space models or continuous-state controllers in simulation. Scaling Active Inference to rich, high-dimensional environments, for example with deep neural networks as generative models, is under active development, and no demonstration of general-purpose competence has been reported. Claims that the framework yields AGI, or that it is more likely to be safe, are hypotheses.

## In this repository

The discrete-time agent used throughout the codebase follows the POMDP formulation described in [[knowledge_base/mathematics/pomdp_framework]], with action selection by [[knowledge_base/mathematics/expected_free_energy]]. Hierarchical structure is covered in [[knowledge_base/cognitive/hierarchical_models]], and planning as inference in [[knowledge_base/cognitive/planning_as_inference]]. The contrast with reward-driven agents is discussed in [[knowledge_base/cognitive/reinforcement_learning]]. Safety questions are collected in [[knowledge_base/free_energy_principle/applications/ai_safety]], and the longer-range discussion sits in [[knowledge_base/research/superintelligence]]. A guided path is in [[active_inference_agi_learning_path]].

## References

- Hutter, M. (2005). *Universal Artificial Intelligence*. Springer.
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.
- Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction*.

## Related Resources

- [[knowledge_base/cognitive/active_inference]]
- [[knowledge_base/cognitive/predictive_processing]]
