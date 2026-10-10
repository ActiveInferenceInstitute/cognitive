---
title: Spatial Memory
description: "Memory for locations, routes, and layouts, covering cognitive maps, place and grid cells, egocentric versus allocentric coding, and path integration, with a generative-model reading of navigation."
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - memory
  - navigation
  - hippocampus
semantic_relations:
  - type: relates
    links:
      - "[[spatial_cognition]]"
      - "[[episodic_memory]]"
      - "[[social_insect_cognition]]"
      - "[[epistemic_foraging]]"
---

# Spatial Memory

## Overview
Spatial memory is the storage and retrieval of information about where things are and how to get between places. It supports navigation, object-location recall, and the recall of events in their settings. Tolman's work on rats introduced the idea of a cognitive map; later work identified neural representations that appear to implement one.

## Core elements
- **Reference frames.** Egocentric representations code location relative to the body (left, ahead); allocentric representations code it relative to the environment. Parietal cortex is associated with the former and the hippocampal formation with the latter.
- **Place cells.** Hippocampal neurons fire when an animal is in a particular location (O'Keefe and Dostrovsky).
- **Grid cells.** Medial entorhinal neurons fire at multiple locations arranged in a hexagonal lattice (Hafting and colleagues). Head-direction and boundary cells add orientation and geometry.
- **Path integration.** Position is updated from self-motion cues alone. Desert ants navigate home along a straight line after convoluted outward paths using this strategy, supplemented by landmark memory ([[social_insect_cognition]]).
- **Strategies.** Landmark-based, route-based, and survey-based (map-like) strategies coexist; the hippocampus is linked to flexible map-based behaviour and the striatum to habitual route following.
- **Human findings.** Hippocampal damage impairs place memory, and enlarged posterior hippocampi are reported in London taxi drivers (a correlational finding).

## Role in Active Inference
Spatial memory is naturally cast as a learned generative model of the environment: hidden states are locations, transitions encode movement, and observations are landmark cues. Navigation then becomes planning under this model, with epistemic value driving exploration of unvisited places ([[epistemic_foraging]], [[planning_as_inference]]). The hippocampal formation has been proposed to supply a spatial-temporal scaffold for such inference, though this is a theoretical proposal rather than an established mapping. Related successor-representation and grid-cell models make similar predictions without the free-energy framing.

## Related concepts
- [[spatial_cognition]] - the wider study of space in the mind
- [[episodic_memory]] - events remembered with their place
- [[state_space_abstraction]] - reducing a map to useful states
- [[statistical_learning]] - acquiring layouts from experience
- [[spatial_attention]] - selecting locations in the present

## References
- Tolman, E. C. (1948). Cognitive maps in rats and men. Psychological Review.
- O'Keefe, J., & Dostrovsky, J. (1971). The hippocampus as a spatial map. Preliminary evidence from unit activity in the freely-moving rat. Brain Research.
- Hafting, T., Fyhn, M., Molden, S., Moser, M.-B., & Moser, E. I. (2005). Microstructure of a spatial map in the entorhinal cortex. Nature.
- Maguire, E. A., et al. (2000). Navigation-related structural change in the hippocampi of taxi drivers. Proceedings of the National Academy of Sciences.
