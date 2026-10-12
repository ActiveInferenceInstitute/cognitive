---
title: Visual Cortex
description: The occipital and adjacent cortical areas that process visual information, from retinotopic primary cortex to higher extrastriate areas, and their central role in predictive coding models of perception.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - vision
  - cortex
  - predictive_coding
semantic_relations:
  - type: relates
    links:
      - "[[visual_perception]]"
      - "[[visual_neuroscience]]"
      - "[[predictive_coding]]"
      - "[[hierarchical_processing]]"
      - "[[edge_detection]]"
wikidata: Q75785
---

# Visual Cortex

## Overview
The visual cortex is the part of the cerebral cortex that receives and processes visual information. Its first stage, primary visual cortex (V1, striate cortex, Brodmann area 17), lies along the calcarine sulcus in the occipital lobe and receives input from the retina via the lateral geniculate nucleus of the thalamus. Surrounding extrastriate areas, including V2, V3, V4 and the motion-sensitive area MT, form a hierarchy of increasingly specialised representations.

## Organisation and function
- **Retinotopy.** Neighbouring points in the visual field map to neighbouring points in V1, with a disproportionately large area devoted to the fovea.
- **Receptive fields.** Hubel and Wiesel (1962) found that V1 neurons respond selectively to oriented edges in particular locations. They distinguished simple cells, with distinct on and off subregions, from complex cells, which respond to an oriented edge across positions. Neurons are arranged in columns for orientation and ocular dominance.
- **Hierarchy and streams.** Felleman and Van Essen (1991) compiled anatomical evidence for a hierarchy of visual areas linked by feedforward and feedback connections. A common summary, proposed by Ungerleider and Mishkin, distinguishes a ventral stream toward temporal cortex for object identity and a dorsal stream toward parietal cortex for spatial relations and action, though later work stresses substantial interaction between them.
- **Feedback.** Feedback projections outnumber feedforward ones in many pathways and modulate responses according to attention, context and expectation.

## Relation to Active Inference
Visual cortex is the best-known testing ground for predictive coding. Rao and Ballard (1999) showed that a hierarchical network in which higher areas predict the activity of lower ones, and lower areas send forward only the residual error, reproduced extra-classical receptive field effects such as end-stopping. This is an influential model that links the cortical hierarchy to the message-passing scheme used in Active Inference (see [[predictive_coding]] and [[free_energy_principle]]). Evidence for the strict form of the scheme is mixed, since error and prediction signals have proven difficult to isolate in neural recordings, and it remains an active research question.

## Related concepts
- [[visual_perception]] — the perceptual functions that the cortex supports
- [[visual_neuroscience]] — the wider field studying the visual system
- [[predictive_coding]] — the computational theory most associated with its hierarchy
- [[hierarchical_processing]] — the layered organisation of visual areas
- [[edge_detection]] — early feature extraction performed by V1
- [[motion_perception]] — processing in area MT and associated dorsal-stream regions

## References
- Hubel, D. H., & Wiesel, T. N. (1962). Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. Journal of Physiology, 160, 106-154.
- Felleman, D. J., & Van Essen, D. C. (1991). Distributed hierarchical processing in the primate cerebral cortex. Cerebral Cortex, 1, 1-47.
- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: A functional interpretation of some extra-classical receptive-field effects. Nature Neuroscience, 2, 79-87.
