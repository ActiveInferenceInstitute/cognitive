---
title: Visual Working Memory
description: The limited-capacity system that holds visual information in mind over short delays, its capacity limits, competing slot and resource models, neural substrates, and links to belief precision.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - working_memory
  - vision
  - capacity_limits
semantic_relations:
  - type: relates
    links:
      - "[[working_memory]]"
      - "[[visuospatial_sketchpad]]"
      - "[[visual_perception]]"
      - "[[capacity_limits]]"
      - "[[iconic_memory]]"
wikidata: Q97700035
---

# Visual Working Memory

## Overview
Visual working memory is the capacity to retain and manipulate visual information, such as colours, shapes, orientations and locations, over a delay of seconds after it has disappeared from view. It is distinguished from brief sensory storage ([[iconic_memory]]), which decays within a fraction of a second, and from long-term memory. In the multicomponent model of Baddeley and Hitch (1974) it corresponds to the visuospatial sketchpad.

## Capacity and models
- **Capacity limit.** Luck and Vogel (1997) found that observers can hold about three to four objects in a change-detection task, and that capacity was determined by the number of objects rather than by the number of features each object carried. Cowan (2001) reached a similar estimate across tasks and modalities.
- **Slot models.** These models hold that memory consists of a small number of discrete slots, each of which stores one item at fixed resolution, with items beyond the limit not stored.
- **Resource models.** These models hold that a continuous resource is divided among all items, so that precision decreases as the set size grows (Bays & Husain, 2008). Evidence from continuous-report tasks, in which the error distribution widens with load, favours variants of resource models, and hybrid proposals also exist (Ma et al., 2014).
- **Neural substrates.** Delay-period activity in prefrontal and parietal cortex has long been linked to maintenance. Decoding studies show that information about remembered stimuli can also be read out from early visual cortex, which suggests that sensory areas take part in storage (Harrison & Tong, 2009).

## Relation to Active Inference
In Active Inference, holding a visual item in mind can be described as maintaining a belief about a hidden state across a delay without direct sensory support. A resource view maps naturally to the precision of that belief, which would be spread across items, so that the quality of each representation declines as more are held. This mapping is a conceptual one. Active Inference models of working memory exist, but there is no consensus implementation, and such models have not yet been used to settle the slot-versus-resource question.

## Related concepts
- [[working_memory]] — the general system of which this is a part
- [[visuospatial_sketchpad]] — the corresponding component of the Baddeley model
- [[visual_perception]] — the processing that supplies the content
- [[capacity_limits]] — the constraints on how much can be retained
- [[iconic_memory]] — the brief sensory store that precedes working memory
- [[precision_weighting]] — the Active Inference notion mapped to resource

## References
- Baddeley, A. D., & Hitch, G. (1974). Working memory. In G. H. Bower (Ed.), The Psychology of Learning and Motivation, vol. 8, 47-89. New York: Academic Press.
- Luck, S. J., & Vogel, E. K. (1997). The capacity of visual working memory for features and conjunctions. Nature, 390, 279-281.
- Cowan, N. (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. Behavioral and Brain Sciences, 24, 87-114.
- Bays, P. M., & Husain, M. (2008). Dynamic shifts of limited working memory resources in human vision. Science, 321, 851-854.
- Harrison, S. A., & Tong, F. (2009). Decoding reveals the contents of visual working memory in early visual areas. Nature, 458, 632-635.
- Ma, W. J., Husain, M., & Bays, P. M. (2014). Changing concepts of working memory. Nature Neuroscience, 17, 347-356.
