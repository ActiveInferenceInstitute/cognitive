---
title: Context Effects
description: How surrounding information, situation and prior experience change perception, memory and judgement of a target item, with classic examples and the Bayesian account of why context helps.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - perception
  - memory
  - context
semantic_relations:
  - type: relates
    links:
      - "[[perceptual_inference]]"
      - "[[prior_beliefs]]"
      - "[[predictive_processing]]"
---

# Context Effects

## Overview
Context effects are changes in how a stimulus is perceived, remembered or judged that depend on what surrounds it in space, time, meaning or situation. They are pervasive: the same physical input is interpreted differently in different circumstances, and this is generally adaptive because context is informative about what is likely.

## Examples
- **Perception**: letters are recognised more accurately in words than in isolation or in non-words, which McClelland and Rumelhart (1981) modelled with interactive activation, where word-level units feed back to letter-level units. Objects are identified faster and more accurately when consistent with their scene, and Bar (2004) reviewed how contextual associations generate expectations that facilitate recognition. Simultaneous contrast and lightness illusions show that perceived brightness or colour depends on neighbouring regions (see [[brightness_perception]], [[color_processing]]).
- **Language**: word meaning, phonemes and syntactic structure are disambiguated by sentence and discourse context (see [[language_processing]], [[constraint_satisfaction]]).
- **Memory**: Godden and Baddeley (1975) found that divers recalled word lists better in the environment (land or underwater) in which they learned them, an instance of context-dependent memory. Related effects include state-dependent memory and encoding specificity.
- **Judgement and decision**: preferences shift with the choice set, as in attraction and compromise effects (see [[decision_making]]).
- **Extinction and clinical phenomena**: renewal of extinguished responses after a context change (see [[behavioral_therapy]]).

## Role in Active Inference
A generative model integrates prior expectations with sensory evidence, and context is the source of many priors. Context sets which hidden causes are probable, so ambiguous input is resolved toward contextually likely interpretations and processing is faster when context is predictive (see [[perceptual_inference]], [[prior_beliefs]]). In hierarchical models, higher levels carry contextual states that supply priors to lower levels (see [[hierarchical_inference]], [[context_modulation]]). Illusions that depend on context are then interpreted as optimal inference under assumptions that fail in the unusual case. This is a general principle that many Bayesian perception models share, not a prediction unique to Active Inference, and some context effects (for example repulsive contrast effects) need additional assumptions to be explained.

## Related concepts
- [[predictive_processing]] — the framework in which context supplies expectations
- [[cross_modal_perception]] — context from other senses
- [[episodic_memory]] — retrieval depends on the match to encoding context
- [[metacognition]] — awareness of how context biases judgement

## References
- Godden, D. R., & Baddeley, A. D. (1975). Context-dependent memory in two natural environments: On land and underwater. British Journal of Psychology, 66, 325-331.
- McClelland, J. L., & Rumelhart, D. E. (1981). An interactive activation model of context effects in letter perception: Part 1. An account of basic findings. Psychological Review, 88, 375-407.
- Bar, M. (2004). Visual objects in context. Nature Reviews Neuroscience, 5, 617-629.
