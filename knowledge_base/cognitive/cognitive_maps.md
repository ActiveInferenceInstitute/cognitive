---
title: Cognitive Maps
description: Internal representations of spatial layout and, more generally, of relations among states, their origin in Tolman's work, hippocampal mechanisms, and links to generative models in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - spatial_cognition
  - hippocampus
  - representation
semantic_relations:
  - type: relates
    links:
      - "[[spatial_cognition]]"
      - "[[hippocampus]]"
      - "[[grid_cells]]"
      - "[[spatial_memory]]"
      - "[[generative_model]]"
---

# Cognitive Maps

## Overview
A cognitive map is an internal representation of the spatial layout of an environment, or more generally of the relations among states of a task, that supports flexible navigation and inference. The term comes from Tolman (1948), who argued from rat maze experiments on latent learning and shortcut behaviour that animals acquire a map-like structure rather than only stimulus-response chains. The idea contrasts with route-based strategies that store fixed sequences of turns.

## Neural basis
O'Keefe and Nadel (1978) proposed that the hippocampus implements a cognitive map. The proposal rested on place cells, hippocampal neurons that fire when an animal occupies a particular location. Later discoveries added head-direction cells, border or boundary cells and the periodic firing of [[grid_cells]] in the medial entorhinal cortex, which together provide candidate components for a map and for path integration. Humans show comparable signals in imaging and in recordings from patients, and damage to hippocampal circuits impairs navigation and some forms of relational memory. Whether the map is a metric Euclidean representation or a topological graph of places and links remains debated, and behavioural evidence shows systematic distortions of recalled layouts.

## Generalisation beyond space
Behrens and colleagues (2018) argued that the same machinery represents non-spatial relational structure, such as social hierarchies or task states, enabling inference about unobserved transitions and generalisation to new problems. On this view a cognitive map is a learned structure over a state space, and the relevant question is how that structure is acquired and reused.

## Relation to Active Inference
The link is conceptual. A cognitive map can be read as part of a [[generative_model]]: the transition structure over hidden states of an environment, together with an observation mapping, supports planning by simulating the consequences of policies. Active Inference agents navigating a maze are standard simulations, and hippocampal-entorhinal function has been interpreted as inference over such structure. These readings are active research hypotheses and are not unique to Active Inference, since model-based reinforcement learning uses closely related constructs.

## Related concepts
- [[spatial_cognition]] — the broader study of space and navigation in the mind
- [[hippocampus]] — the structure most associated with map-like representation
- [[grid_cells]] — entorhinal neurons with periodic spatial firing
- [[spatial_memory]] — memory for locations and layouts
- [[generative_model]] — the Active Inference counterpart of a map
- [[entorhinal_cortex]] — the interface between neocortex and hippocampus

## References
- Tolman, E. C. (1948). Cognitive maps in rats and men. Psychological Review, 55, 189-208.
- O'Keefe, J., & Nadel, L. (1978). The Hippocampus as a Cognitive Map. Oxford University Press.
- Behrens, T. E. J., et al. (2018). What is a cognitive map? Organizing knowledge for flexible behavior. Neuron, 100, 490-509.
