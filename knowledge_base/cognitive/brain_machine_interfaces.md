---
title: Brain-Machine Interfaces
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - neurotechnology
  - decoding
  - closed-loop
semantic_relations:
  - type: relates
    links:
      - "[[neural_decoding]]"
      - "[[neuroprosthetics]]"
      - "[[motor_control]]"
aliases:
  - neural_interfaces
  - brain_computer_interface
  - brain_computer_interfaces
  - brain_interfaces
  - brain_machine
wikidata: Q897410
---

# Brain-Machine Interfaces

## Overview
A brain-machine interface (BMI), also called a brain-computer interface (BCI), is a system that records neural activity, translates it into commands for an external device, and often returns feedback to the user. The aims include restoring communication or movement after paralysis and studying how the brain learns to control a new effector.

## Mechanism
The signal chain has four parts: acquisition, feature extraction, decoding, and output with feedback. Acquisition ranges from non-invasive electroencephalography (EEG), through electrocorticography on the cortical surface, to penetrating microelectrode arrays that record spiking activity and local field potentials. Invasive recordings generally provide higher information rates but carry surgical risk and long-term signal stability problems.

Decoders map neural features to control variables. Early motor BMIs used linear filters and population vectors; Kalman filters, which combine a movement model with a noisy neural observation model, are common, and recent work uses recurrent neural networks. Hochberg and colleagues demonstrated in 2006 that a person with tetraplegia could control a computer cursor and simple robotic devices from motor cortical ensemble activity; later work extended this to reach and grasp with a robotic arm. Wolpaw and colleagues review non-invasive BCIs for communication, such as those using the P300 or sensorimotor rhythms.

BMIs are closed-loop systems, which sets them apart from offline decoding. The user observes the output and adjusts neural activity, and the decoder may be adapted in parallel, so performance depends on co-adaptation between brain and algorithm. Users can improve with practice, consistent with the brain learning a new mapping.

## Role in Active Inference
A BMI is a clear example of an agent acting through a new, artificial sensorimotor loop. In active-inference terms, the user must learn a generative model of how neural activity maps to cursor or limb motion and select neural "actions" that fulfil preferred outcomes. The framework offers a way to think about adaptation, sense of agency over the device, and the role of feedback precision. Concrete BMI designs based on active inference are exploratory, and the dominant practical methods are standard signal-processing and machine-learning decoders. The link is therefore interpretive.

## Related concepts
- [[neural_decoding]] — the algorithms that translate activity into commands
- [[neuroprosthetics]] — devices BMIs control or that provide sensory feedback
- [[motor_control]] — the natural system a motor BMI substitutes for
- [[population_coding]] — representation exploited by decoders
- [[sensorimotor_coordination]] — loop that the interface must replace
- [[neural_plasticity]] — basis of user learning

## References
- Wolpaw, J. R., Birbaumer, N., McFarland, D. J., Pfurtscheller, G., & Vaughan, T. M. (2002). Brain-computer interfaces for communication and control. Clinical Neurophysiology.
- Hochberg, L. R., et al. (2006). Neuronal ensemble control of prosthetic devices by a human with tetraplegia. Nature.
