---
title: What Stream
description: The ventral visual pathway for object identification, its hierarchy and category-selective regions, contrast with the dorsal pathway, and its role in predictive coding and in Active Inference models.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - visual_system
  - ventral_stream
  - object_recognition
semantic_relations:
  - type: relates
    links:
      - "[[ventral_stream]]"
      - "[[where_stream]]"
      - "[[object_recognition]]"
      - "[[visual_cortex]]"
      - "[[predictive_coding]]"
      - "[[adaptive_resonance_theory]]"
---

# What Stream

## Overview
The what stream is the visual processing pathway that runs from primary visual cortex through V2 and V4 to inferotemporal cortex and is specialised for identifying what is seen. Ungerleider and Mishkin (1982) distinguished it from a dorsal "where" pathway after lesion studies in monkeys: damage to the inferior temporal cortex impaired object discrimination, whereas damage to posterior parietal cortex impaired spatial judgements (Mishkin, Ungerleider and Macko, 1983). It is largely the same anatomical system as the [[ventral_stream]], and the "what" label emphasises its function. Goodale and Milner (1992) later reinterpreted the distinction as vision for perception versus vision for action (see [[where_stream]]).

## Organisation
- **Hierarchy.** Along the pathway receptive fields grow larger, and neurons respond to more complex features, such as contours, shape parts and whole objects. Responses become more tolerant to changes in position and size while remaining selective for identity (DiCarlo, Zoccolan and Rust, 2012).
- **Category-selective regions.** In humans, areas of ventral cortex respond preferentially to faces (the fusiform face area; Kanwisher, McDermott and Chun, 1997), to scenes, to bodies and to printed words. How far these regions are specialised modules is debated.
- **Feedback.** Anatomical connections run in both directions, and recurrent processing contributes to recognition under difficult conditions.

## Role in predictive processing and Active Inference
Rao and Ballard (1999) proposed that visual cortex implements hierarchical predictive coding, in which higher levels send predictions of lower-level activity and lower levels send prediction errors, and they used it to explain extra-classical receptive field effects. The ventral pathway is the standard example of this hierarchy. In Active Inference models of visual foraging, hidden states are factorised into identity ("what") and location ("where"), mirroring the two pathways, so that the model infers what an object is and where it is separately (Mirza et al., 2016). The factorisation is a modelling choice inspired by anatomy and does not follow from the theory. In adaptive resonance theory the "what" system performs learned category recognition, complementing a spatial "where" system.

## Related concepts
- [[ventral_stream]] — the pathway described anatomically
- [[where_stream]] — the dorsal counterpart for spatial processing
- [[object_recognition]] — the main function of the stream
- [[visual_cortex]] — the cortical hierarchy that contains it
- [[predictive_coding]] — the hierarchical scheme proposed for its processing
- [[adaptive_resonance_theory]] — a theory that pairs "what" and "where" systems

## References
- Mishkin, M., Ungerleider, L. G., & Macko, K. A. (1983). Object vision and spatial vision: two cortical pathways. Trends in Neurosciences, 6, 414-417.
- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature Neuroscience, 2, 79-87.
- DiCarlo, J. J., Zoccolan, D., & Rust, N. C. (2012). How does the brain solve visual object recognition? Neuron, 73, 415-434.
- Mirza, M. B., Adams, R. A., Mathys, C. D., & Friston, K. J. (2016). Scene construction, visual foraging, and active inference. Frontiers in Computational Neuroscience, 10, 56.
- Ungerleider, L. G., & Mishkin, M. (1982). Two cortical visual systems. In D. J. Ingle, M. A. Goodale, & R. J. W. Mansfield (Eds.), Analysis of Visual Behavior (pp. 549-586). MIT Press.
- Goodale, M. A., & Milner, A. D. (1992). Separate visual pathways for perception and action. Trends in Neurosciences, 15, 20-25.
- Kanwisher, N., McDermott, J., & Chun, M. M. (1997). The fusiform face area: A module in human extrastriate cortex specialized for face perception. Journal of Neuroscience, 17, 4302-4311.
