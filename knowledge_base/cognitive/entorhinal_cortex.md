---
title: Entorhinal Cortex
description: The cortical region that forms the main gateway between neocortex and hippocampus, its medial and lateral divisions, grid cells, and its role in memory, navigation and early Alzheimer's pathology.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - hippocampus
  - spatial_cognition
  - memory
semantic_relations:
  - type: relates
    links:
      - "[[hippocampus]]"
      - "[[grid_cells]]"
      - "[[dentate_gyrus]]"
      - "[[medial_temporal_lobe]]"
      - "[[spatial_memory]]"
---

# Entorhinal Cortex

## Overview
The entorhinal cortex is a region of the medial temporal lobe, corresponding mainly to Brodmann areas 28 and 34, that serves as the principal interface between the hippocampus and the neocortex. It receives convergent input from association, perirhinal and parahippocampal cortices and returns hippocampal output to those areas (van Strien, Cappaert and Witter, 2009). Its position makes it essential for transmitting information into and out of the hippocampus.

## Connectivity and divisions
Superficial layers, mainly layer II, project to the [[dentate_gyrus]] and CA3 through the perforant path, and layer III projects to CA1 and the subiculum. Deep layers receive hippocampal output and relay it back to cortex. The region has two main subdivisions, the medial and the lateral entorhinal cortex, which differ in their inputs. The medial division is more strongly connected with spatial and visuospatial areas, and the lateral division with perirhinal and object-related areas. Correspondingly, the medial entorhinal cortex contains grid cells (Hafting et al., 2005), alongside head-direction and boundary-related cells, whereas lateral entorhinal cells are described as carrying information about objects, context and elapsed time, although this division is a broad generalisation.

## Clinical relevance
Neurofibrillary tangle pathology in Alzheimer's disease appears early in the transentorhinal and entorhinal regions before spreading, and entorhinal atrophy is used as an early imaging marker. This is consistent with the early memory and navigation difficulties associated with the disease (see [[alzheimers_disease]]).

## Relation to Active Inference
The link is indirect. Computational models treat the hippocampal-entorhinal system as learning a structured representation of relations between states, in which grid-like codes act as a basis for generalising over space and over abstract relationships (Whittington et al., 2020). Active Inference describes spatial navigation as inference and planning under a generative model of the environment, and the entorhinal cortex is a candidate substrate for the state representations such a model needs. This is an active research programme, not a derivation, and the leading entorhinal models are not formulated within the free-energy framework.

## Related concepts
- [[hippocampus]] — the structure that the entorhinal cortex gates
- [[grid_cells]] — periodic spatial neurons of the medial division
- [[dentate_gyrus]] — target of the perforant path
- [[medial_temporal_lobe]] — the wider memory system
- [[spatial_memory]] — memory for places and routes
- [[cognitive_maps]] — map-like representations it is proposed to support

## References
- Hafting, T., Fyhn, M., Molden, S., Moser, M.-B., & Moser, E. I. (2005). Microstructure of a spatial map in the entorhinal cortex. Nature, 436, 801-806.
- van Strien, N. M., Cappaert, N. L. M., & Witter, M. P. (2009). The anatomy of memory: an interactive overview of the parahippocampal-hippocampal network. Nature Reviews Neuroscience, 10, 272-282.
- Whittington, J. C. R., et al. (2020). The Tolman-Eichenbaum Machine: unifying space and relational memory through generalization in the hippocampal formation. Cell, 183, 1249-1263.
