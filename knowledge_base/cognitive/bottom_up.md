---
title: Bottom-Up Processing
description: Stimulus-driven, data-driven processing that proceeds from sensory input toward abstract representations, contrasted with top-down influence, and its reading as ascending prediction error.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - perception
  - attention
  - hierarchy
semantic_relations:
  - type: relates
    links:
      - "[[predictive_coding]]"
      - "[[hierarchical_processing]]"
      - "[[exogenous_attention]]"
aliases:
  - bottom_up_processing
  - bottom_up_control
---

# Bottom-Up Processing

## Overview
Bottom-up (data-driven) processing is information flow that begins with sensory input and proceeds toward more abstract representations, with the stimulus determining the outcome. It is contrasted with top-down (concept-driven) processing, in which expectations, goals and context shape interpretation. In the emergence literature the same phrase also means building system-level structure from interacting components (see [[emergence_self_organization]]); this page treats the perceptual and attentional sense.

## Core claims
- **Feedforward sweep**: in vision, a first pass of activity through the ventral stream can support rapid categorisation of simple scenes, consistent with a largely feedforward stage (see [[visual_perception]], [[hierarchical_processing]]).
- **Stimulus-driven attention**: salient or abrupt stimuli capture attention independently of goals. Itti and Koch (2001) modelled this with a saliency map combining feature contrasts, and Corbetta and Shulman (2002) distinguished a ventral frontoparietal circuit for stimulus-driven reorienting from a dorsal circuit for goal-directed selection (see [[exogenous_attention]]).
- **Interaction with top-down signals**: most behaviour reflects both. Reading and speech comprehension combine letter or acoustic evidence with lexical and sentence context (see [[language_processing]], [[context_effects]]). The two directions are not separate modules but parts of recurrent processing.
- **Limits**: purely bottom-up accounts struggle with ambiguity, occlusion and noise, where input underdetermines the interpretation.

## Role in Active Inference
In hierarchical predictive coding the bottom-up pathway carries prediction errors, not raw sensory features: each level sends upward the part of its input not explained by predictions from the level above (Rao and Ballard, 1999). Bottom-up influence is thus the ascending error signal, weighted by its precision, while top-down influence is the descending prediction (see [[prediction_error]], [[precision_weighting]]). The balance between the two is governed by relative precision: high sensory precision lets input dominate, and high prior precision lets expectation dominate. In the canonical microcircuit proposal, forward connections arise mainly from superficial layers and backward connections from deep layers (Bastos et al., 2012), though the empirical mapping is incomplete. The framework reframes the dichotomy as a precision balance and so does not treat purely bottom-up or purely top-down processing as distinct modes.

## Related concepts
- [[perceptual_inference]] — inference that combines both directions
- [[feature_extraction]] — a classic bottom-up stage
- [[attention_mechanisms]] — stimulus-driven and goal-driven selection
- [[sensory_processing]] — early stages feeding the bottom-up sweep

## References
- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature Neuroscience, 2, 79-87.
- Itti, L., & Koch, C. (2001). Computational modelling of visual attention. Nature Reviews Neuroscience, 2, 194-203.
- Corbetta, M., & Shulman, G. L. (2002). Control of goal-directed and stimulus-driven attention in the brain. Nature Reviews Neuroscience, 3, 201-215.
- Bastos, A. M., et al. (2012). Canonical microcircuits for predictive coding. Neuron, 76, 695-711.
