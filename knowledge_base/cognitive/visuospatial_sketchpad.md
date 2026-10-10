---
title: Visuospatial Sketchpad
description: "The working-memory component that temporarily holds and manipulates visual and spatial information, its visual cache and inner scribe subdivision, capacity limits, and a predictive-inference reading."
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - working-memory
  - visual
  - spatial
semantic_relations:
  - type: relates
    links:
      - "[[working_memory]]"
      - "[[phonological_loop]]"
      - "[[spatial_memory]]"
      - "[[visual_perception]]"
---

# Visuospatial Sketchpad

## Overview
The visuospatial sketchpad is one of the subsystems of the multicomponent working-memory model of Baddeley and Hitch. It maintains and manipulates visual and spatial material over seconds, in parallel with the [[phonological_loop]] for verbal material, and both are coordinated by a central executive. It supports tasks such as imagining a route, mentally rotating a shape, or remembering where objects lie on a table.

## Structure and evidence
- **Subcomponents.** Logie proposed a visual cache, a passive store of form and colour, and an inner scribe, which handles spatial and movement sequences and is linked to rehearsal and to motor planning.
- **Dissociations.** Dual-task studies show that spatial tapping interferes more with spatial memory tasks and visual noise more with visual pattern memory. Patient data show selective deficits in one or the other.
- **Capacity.** Visual working memory holds a small number of objects, around three to four, with precision declining as more are stored; the best account of this limit (slots versus a shared resource) remains debated (Luck and Vogel reported the object-based limit).
- **Mental imagery and rotation.** Time to judge whether two shapes match increases with their angular difference, suggesting an analogue transformation process (Shepard and Metzler).
- **Neural basis.** Sustained activity in occipital, parietal, and prefrontal regions during delays is linked to maintenance, with spatial and object contents associated with dorsal and ventral pathway regions respectively ([[spatial_cognition]]).
- **Relation to other models.** Some accounts treat visual working memory as sustained activation of the same sensory representations used for perception, reducing the need for a separate store.

## Role in Active Inference
Active Inference treats working memory as the maintenance of beliefs about hidden states in the absence of current input, where the contents persist because the generative model's dynamics and prior expectations hold them. The precision assigned to those beliefs would set how well they resist interference and decay, and attention to remembered items would correspond to increased precision ([[precision_weighting]], [[belief_updating]]). Simulations along these lines exist, but the mapping to the multicomponent model is conceptual and the subdivision into cache and scribe has not been derived from the framework.

## Related concepts
- [[working_memory]] - the system of which it is a component
- [[spatial_memory]] - long-term spatial representations
- [[visual_attention]] - selection of what is held
- [[perceptual_inference]] - the same representations used in perception
- [[executive_functions]] - control processes acting on the sketchpad

## References
- Baddeley, A. D., & Hitch, G. (1974). Working memory. In G. H. Bower (Ed.), The Psychology of Learning and Motivation, Vol. 8. Academic Press.
- Logie, R. H. (1995). Visuo-spatial Working Memory. Lawrence Erlbaum.
- Shepard, R. N., & Metzler, J. (1971). Mental rotation of three-dimensional objects. Science.
- Luck, S. J., & Vogel, E. K. (1997). The capacity of visual working memory for features and conjunctions. Nature.
