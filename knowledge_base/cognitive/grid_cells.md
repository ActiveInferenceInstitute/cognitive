---
title: Grid Cells
description: Neurons in the medial entorhinal cortex that fire at the vertices of a periodic triangular lattice as an animal moves through space, their organisation, and their role in path integration and relational coding.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - spatial_cognition
  - entorhinal_cortex
  - neural_coding
semantic_relations:
  - type: relates
    links:
      - "[[entorhinal_cortex]]"
      - "[[hippocampus]]"
      - "[[cognitive_maps]]"
      - "[[spatial_memory]]"
      - "[[attractor_dynamics]]"
---

# Grid Cells

## Overview
Grid cells are neurons that fire when an animal is at any of many locations arranged in a regular, repeating triangular (hexagonal) pattern covering the whole environment. They were reported in the medial entorhinal cortex of rats by Hafting and colleagues (2005), and form part of the positional system for which John O'Keefe, May-Britt Moser and Edvard Moser received the 2014 Nobel Prize in Physiology or Medicine. The firing fields of an individual cell are defined by spacing, orientation and spatial phase.

## Organisation
Grid cells are arranged in modules. Within a module the cells share spacing and orientation but differ in phase, and spacing increases in discrete steps along the dorsal-to-ventral axis of the medial entorhinal cortex (Stensola et al., 2012). Together with head-direction cells, border cells and speed-sensitive cells in the same region, they provide a periodic code for position that can represent a large range of locations with a modest number of cells (Moser, Kropff and Moser, 2008). Grid-like signals have also been detected in humans with functional imaging during virtual navigation.

## Proposed function
The leading idea is that grid cells support path integration, the updating of an estimated position from self-motion cues such as speed and heading. Continuous attractor network models reproduce the lattice and its updating by movement (see [[attractor_dynamics]]). Grid cells interact with place cells in the hippocampus, but how place fields depend on grid input is unsettled, since place fields persist in some conditions after grid disruption. Evidence also indicates that grid-like coding extends to non-spatial dimensions, such as abstract conceptual spaces (Constantinescu, O'Reilly and Behrens, 2016), which suggests a general coding scheme for relational structure.

## Relation to Active Inference
The connection is an area of current research and not a derivation. Navigation can be treated as inference about location under a generative model, with self-motion providing a transition model and sensory landmarks providing observations, and grid codes are one candidate for the underlying state representation. The leading models of grid cells have been developed in dynamical-systems and normative-coding frameworks, and are not specific to free-energy theories.

## Related concepts
- [[entorhinal_cortex]] — the location of grid cells
- [[hippocampus]] — the partner structure containing place cells
- [[cognitive_maps]] — the representational concept grid cells help realise
- [[spatial_memory]] — memory for locations that depends on this system
- [[attractor_dynamics]] — the dynamics used to model grid patterns

## References
- Hafting, T., Fyhn, M., Molden, S., Moser, M.-B., & Moser, E. I. (2005). Microstructure of a spatial map in the entorhinal cortex. Nature, 436, 801-806.
- Moser, E. I., Kropff, E., & Moser, M.-B. (2008). Place cells, grid cells, and the brain's spatial representation system. Annual Review of Neuroscience, 31, 69-89.
- Stensola, H., et al. (2012). The entorhinal grid map is discretized. Nature, 492, 72-78.
- Constantinescu, A. O., O'Reilly, J. X., & Behrens, T. E. J. (2016). Organizing conceptual knowledge in humans with a gridlike code. Science, 352, 1464-1468.
