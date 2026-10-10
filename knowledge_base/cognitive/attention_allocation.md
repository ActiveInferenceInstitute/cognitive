---
title: Attention Allocation
description: How a limited processing budget is distributed across stimuli, locations, features and tasks, and how Active Inference casts the allocation as precision weighting and expected information gain.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - attention
  - resource-allocation
  - precision
semantic_relations:
  - type: relates
    links:
      - "[[precision_weighting]]"
      - "[[attention_mechanisms]]"
      - "[[active_inference]]"
---

# Attention Allocation

## Overview
Attention allocation is the distribution of limited processing resources across competing inputs and tasks. Kahneman (1973) framed attention as a capacity that can be divided among concurrent activities according to momentary demands and policies; later work separated spatial, feature-based and object-based selection (see [[spatial_attention]], [[feature_based_attention]], [[object_based_attention]]). The question of allocation is where to spend processing, and when to switch.

## Determinants
Allocation is shaped by three broad influences that interact:
- **Stimulus-driven factors**: salience, abrupt onsets and novelty (see [[exogenous_attention]]).
- **Goal-driven factors**: task set, expected reward and instructions (see [[endogenous_attention]]).
- **Learned regularities**: statistical expectations about where and when relevant events occur, which bias allocation without explicit intent.

Dual-task and divided-attention experiments show that allocation is graded rather than all-or-none, and that performance trades off between tasks as demand changes (see [[divided_attention]]). Costs of switching between targets add a time penalty to reallocation (see [[task_switching]]).

## Role in Active Inference
Active Inference offers two connected accounts. First, in perception, attention is the optimisation of precision: prediction errors from channels expected to be reliable receive higher gain, so they influence belief updates more (Feldman and Friston, 2010; see [[precision_weighting]]). Allocation then corresponds to setting the precision of sensory channels, feature dimensions or hierarchical levels, with neuromodulators proposed to implement the gain (see [[cholinergic_system]], [[dopamine_system]]). Second, in action, the choice of where to look or what to sample is a policy chosen by minimising expected free energy, where the ambiguity and information-gain terms favour sampling outcomes that most reduce uncertainty about hidden states (see [[epistemic_foraging]], [[expected_free_energy]]). Parr and Friston (2019) discuss how these accounts relate to the distinction between attention and salience.

These are theoretical proposals with supporting simulations and some neuroimaging and pharmacological studies; they are not the only account of attention, and precision as a unifying construct remains debated.

## Related concepts
- [[attentional_control]] — the control processes that set the allocation policy
- [[resource_management]] — allocation as a general resource problem
- [[capacity_limits]] — the limits that make allocation necessary
- [[epistemic_value]] — information gain as a driver of where to sample
- [[selective_attention]] — allocation to some inputs at the expense of others

## References
- Kahneman, D. (1973). Attention and Effort. Prentice-Hall.
- Feldman, H., & Friston, K. J. (2010). Attention, uncertainty, and free-energy. Frontiers in Human Neuroscience, 4, 215.
- Parr, T., & Friston, K. J. (2019). Attention or salience? Current Opinion in Psychology, 29, 1-5.
