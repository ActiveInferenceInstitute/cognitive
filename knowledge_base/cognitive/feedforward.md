---
title: Feedforward Processing
description: Feedforward processing is the one-directional flow of information from lower to higher stages of a neural or computational hierarchy, contrasted with recurrent and feedback processing.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - neural_computation
  - hierarchy
  - vision
semantic_relations:
  - type: relates
    links:
      - "[[recurrent_networks]]"
      - "[[hierarchical_processing]]"
      - "[[predictive_coding]]"
      - "[[neural_computation]]"
aliases:
  - feedforward_processing
---

# Feedforward Processing

## Overview
Feedforward processing is information flow from earlier to later stages of a processing hierarchy, with no connections returning from later stages to earlier ones during the computation. The term applies to artificial networks, to cortical anatomy, and to the initial temporal phase of the response to a stimulus.

## Senses of the term
**Artificial networks.** A feedforward network is a directed acyclic graph of layers, each computing a function of the preceding layer. Multilayer perceptrons and standard convolutional networks are examples. With sufficiently many hidden units, such networks can approximate a wide class of functions. Their output depends only on the current input, so they have no internal memory without extra machinery.

**Cortical anatomy.** In cortical hierarchies, forward connections run from lower to higher areas, with typical origin in superficial layers and termination in layer 4, whereas feedback connections originate in deep layers and terminate outside layer 4 (Felleman and Van Essen, 1991). Feedback connections are at least as numerous as forward ones.

**Temporal dynamics.** Image categorisation can be made by humans within about 150 ms (Thorpe, Fize and Marlot, 1996), which has been taken as evidence that a rapid feedforward sweep can support core object recognition. Lamme and Roelfsema (2000) argued that the feedforward sweep produces unconscious, fast processing of features, while recurrent interactions subserve later perceptual organisation and conscious perception.

Feedforward models of the ventral visual stream predict neural responses and behaviour well on rapid, clear categorisation tasks, but degrade for occluded, cluttered or ambiguous images, where recurrence appears to contribute.

## Role in Active Inference
In predictive coding, the standard Active Inference implementation of perception, the feedforward and feedback paths carry different quantities. Feedback connections convey predictions about activity at lower levels, and feedforward connections convey prediction errors, the residual that remains after prediction (Rao and Ballard, 1999). Perception is therefore intrinsically recurrent: inference iterates between the two directions until errors are minimised. A purely feedforward sweep corresponds to the fast initial estimate with weak top-down contribution. The laminar assignment of prediction errors to superficial pyramidal cells is a proposal for canonical microcircuits, and the evidence for it is mixed.

## Related concepts
- [[recurrent_networks]] — networks with feedback and lateral connections
- [[prediction_error]] — the quantity predicted to ascend the hierarchy
- [[object_recognition]] — the task for which feedforward models are best established
- [[deep_learning]] — modern feedforward architectures
- [[hierarchical_inference]] — inference across levels of a model

## References
- Felleman, D. J., & Van Essen, D. C. (1991). Distributed hierarchical processing in the primate cerebral cortex. Cerebral Cortex.
- Thorpe, S., Fize, D., & Marlot, C. (1996). Speed of processing in the human visual system. Nature.
- Lamme, V. A. F., & Roelfsema, P. R. (2000). The distinct modes of vision offered by feedforward and recurrent processing. Trends in Neurosciences.
- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature Neuroscience.
