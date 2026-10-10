---
title: Abstraction Levels
description: "Degrees of abstraction in cognitive and neural representation, from concrete sensory features to general concepts, and how hierarchical generative models assign slower timescales and broader scope to higher levels."
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - hierarchy
  - representation
  - generative-models
semantic_relations:
  - type: relates
    links:
      - "[[hierarchical_processing]]"
      - "[[hierarchical_models]]"
      - "[[state_space_abstraction]]"
---

# Abstraction Levels

## Overview
Abstraction levels are the ordered degrees of generality at which a system represents the world. A low level holds concrete, specific, rapidly changing content, such as the luminance at one retinal location; a high level holds general, invariant content, such as the category of object or the goal of an action. Moving up a level discards detail and keeps what is shared across many instances. The notion appears in cognitive science, in memory theory and in machine learning.

## Core claims
In perception, cortical hierarchies show an increase in abstraction along their path. Receptive fields grow larger, tuning moves from edges to object categories, and temporal integration windows lengthen: Hasson and colleagues found a hierarchy of temporal receptive windows in human cortex, with sensory areas following moment-to-moment input and higher areas integrating over seconds to minutes ([[hierarchical_processing]], [[visual_perception]]). In memory, schemas and semantic knowledge abstract away from individual episodes ([[semantic_memory]], [[memory_organization]]). In action, a goal such as "make tea" decomposes into sub-goals and then muscle commands ([[action_planning]]).

The same ordering has computational forms. Deep networks learn increasingly abstract features in successive layers ([[deep_learning]]). Abstraction can also be defined as coarse-graining of a state space, mapping many fine-grained states to one coarse state while preserving dynamics relevant to the task ([[state_space_abstraction]], [[bayesian_renormalization]]). Choosing the level is a trade-off: abstract representations generalize and are cheap to compute with, but may lose information needed for fine control.

## Role in Active Inference
Hierarchical generative models are the main place where abstraction levels become formal. Each level generates predictions for the level below and receives prediction errors from it. Kiebel, Daunizeau and Friston proposed that higher levels evolve on slower timescales and so encode more abstract, slowly changing causes, an arrangement they related to the hierarchy of timescales in the brain. Deep temporal models extend this to discrete-state Active Inference, where higher levels set context or goals that constrain the policies available to lower levels ([[hierarchical_inference]], [[hierarchical_models]], [[policy_selection]]). The framework is a normative proposal about how abstraction could be implemented, and the evidence for a strict timescale ordering is partial.

## Related concepts
- [[generative_model]] — the structure that carries the levels
- [[predictive_coding]] — message passing between levels
- [[levels_of_organization]] — a related notion applied to biological systems
- [[cognitive_architecture]] — system-level organization of representational levels

## References
- Hasson, U., Yang, E., Vallines, I., Heeger, D. J., & Rubin, N. (2008). A hierarchy of temporal receptive windows in human cortex. *Journal of Neuroscience*, 28.
- Kiebel, S. J., Daunizeau, J., & Friston, K. J. (2008). A hierarchy of time-scales and the brain. *PLoS Computational Biology*, 4.
- Friston, K. J., Rosch, R., Parr, T., Price, C., & Bowman, H. (2017). Deep temporal models and active inference. *Neuroscience & Biobehavioral Reviews*, 77.
