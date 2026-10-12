---
title: Pattern Completion
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - memory
  - attractor-dynamics
  - hippocampus
semantic_relations:
  - type: relates
    links:
      - "[[memory_systems]]"
      - "[[pattern_recognition]]"
      - "[[generative_model]]"
wikidata: Q7148375
---

# Pattern Completion

## Overview
Pattern completion is the retrieval of a complete stored pattern from a partial or degraded cue. A fragment of a face, a few words of a song, or a smell can recover a whole memory or percept. It is the defining operation of associative (content-addressable) memory.

## Mechanism
In recurrent autoassociative networks, stored patterns are encoded in the connection weights so that they become stable states (attractors) of the network dynamics. Hopfield's network with binary units $s_i=\pm1$ and Hebbian weights $w_{ij}=\frac{1}{N}\sum_\mu \xi_i^\mu \xi_j^\mu$ defines an energy $E=-\tfrac12\sum_{i\ne j} w_{ij}s_is_j$ that asynchronous updates never increase. Starting from a corrupted version of a stored pattern $\xi^\mu$, the dynamics descend to the nearest minimum, which is usually the stored pattern. Capacity is limited: for random patterns it is roughly $0.14N$, beyond which spurious states and retrieval errors appear.

In the brain, Marr's 1971 theory of archicortex and later work (Rolls and colleagues) attribute this role to hippocampal region CA3, whose dense recurrent collaterals make it a candidate autoassociative network. Complementary pattern separation, thought to be carried out by the dentate gyrus, makes similar inputs more distinct before storage. Experimental evidence comes from Nakazawa and colleagues, who found that mice lacking NMDA receptors specifically in CA3 learned normally but were impaired at recalling a spatial memory when given only partial cues. Pattern completion also appears in sensory cortex, where filling-in of occluded contours and missing sounds is attributed to similar recurrent dynamics.

## Role in Active Inference
Completion follows directly from inference under a generative model: given partial observations, the posterior over causes concentrates on the stored hypothesis that best explains them, and the model then predicts the unobserved remainder. Minimizing free energy in this setting plays the role of the energy descent in a Hopfield network; the correspondence is clearest when the model is a Boltzmann-style or other energy-based network, whose recurrent weights act as coupling between hidden states.

The connection is conceptual rather than demonstrated for the hippocampus. Whether CA3 realizes a hierarchical generative model with prediction errors, as opposed to a simple attractor network, is unresolved, and attractor models need not involve free-energy minimization at all.

## Related concepts
- [[memory_systems]] — kinds of memory that rely on completion
- [[pattern_recognition]] — completing a cue versus classifying it
- [[memory_consolidation]] — hippocampal replay and cortical storage
- [[perceptual_inference]] — filling in missing sensory data
- [[generative_model]] — source of the missing-part predictions
- [[neural_dynamics]] — attractor dynamics underlying retrieval

## References
- Marr, D. (1971). Simple memory: a theory for archicortex. Philosophical Transactions of the Royal Society B.
- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences.
- Nakazawa, K., et al. (2002). Requirement for hippocampal CA3 NMDA receptors in associative memory recall. Science.
