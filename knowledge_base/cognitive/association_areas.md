---
title: Association Areas
description: Cortical regions that integrate information across sensory, motor and limbic systems rather than serving a single modality, and their place at the top of the cortical hierarchy.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - neuroscience
  - cortex
  - hierarchy
semantic_relations:
  - type: relates
    links:
      - "[[hierarchical_processing]]"
      - "[[multisensory_integration]]"
      - "[[predictive_processing]]"
---

# Association Areas

## Overview
Association areas are regions of cerebral cortex that are neither primary sensory nor primary motor cortex. They receive converging input from several modalities and from limbic structures, and support functions such as object and scene understanding, attention, language, planning and social cognition. The term is used here in the neuroanatomical sense of Mesulam's scheme, not the psychological sense of associative learning.

## Organisation
Mesulam (1998) distinguished unimodal association cortex, which elaborates one modality (for example inferotemporal cortex for vision), from heteromodal association cortex, which converges several modalities, and from paralimbic and limbic zones that link cortex to memory and emotion. Heteromodal regions include:
- posterior parietal cortex (see [[inferior_parietal]]), involved in spatial attention, sensorimotor transformation and number;
- lateral prefrontal cortex (see [[dorsolateral_prefrontal]]), involved in working memory and rule use;
- medial prefrontal and posterior cingulate cortex (see [[medial_prefrontal]]), involved in self-referential and social processing;
- superior temporal and lateral temporal cortex, involved in language and biological motion.

Anatomically, these regions sit at the far end of feedforward pathways in the hierarchy described by Felleman and Van Essen (1991). Their receptive properties are abstract and multimodal, their responses integrate over longer time windows, and their sulcal and gyral layout is more variable between individuals than primary cortex. Margulies and colleagues (2016) described a principal gradient of macroscale cortical organisation running from primary sensorimotor cortex to transmodal regions that overlap the default-mode network.

## Role in Active Inference
In hierarchical generative models, higher levels encode slower, more abstract and more multimodal causes of sensory input, and send predictions down while receiving prediction errors up (see [[predictive_coding]]). Kiebel, Daunizeau and Friston (2008) proposed that the cortical hierarchy embodies a hierarchy of time scales, with association cortex representing the slowest and most abstract regularities. Mapping specific association areas onto specific levels of a specific model is an active research programme rather than a settled result; the correspondence between a model layer and a cortical area is a modelling choice constrained by anatomy and function.

## Related concepts
- [[hierarchical_processing]] — association cortex forms the upper stages of the hierarchy
- [[multisensory_integration]] — a core function of heteromodal cortex
- [[perception_attention]] — attention networks recruit parietal and prefrontal association cortex
- [[perceptual_inference]] — inference over causes draws on top-down signals from these areas
- [[working_memory]] — maintained through prefrontal and parietal association cortex

## References
- Mesulam, M.-M. (1998). From sensation to cognition. Brain, 121, 1013-1052.
- Felleman, D. J., & Van Essen, D. C. (1991). Distributed hierarchical processing in the primate cerebral cortex. Cerebral Cortex, 1, 1-47.
- Margulies, D. S., et al. (2016). Situating the default-mode network along a principal gradient of macroscale cortical organization. PNAS, 113, 12574-12579.
- Kiebel, S. J., Daunizeau, J., & Friston, K. J. (2008). A hierarchy of time-scales and the brain. PLoS Computational Biology, 4, e1000209.
