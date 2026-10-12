---
title: Perceptual Learning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - learning
  - perception
  - plasticity
semantic_relations:
  - type: relates
    links:
      - "[[neural_plasticity]]"
      - "[[learning_mechanisms]]"
      - "[[perceptual_inference]]"
wikidata: Q7167036
---

# Perceptual Learning

## Overview
Perceptual learning is a lasting improvement in the detection or discrimination of sensory stimuli that results from practice or experience. Examples include radiologists learning to see subtle lesions, wine tasters discriminating flavors, and laboratory training that improves orientation or texture discrimination over days. The change is in perception itself, not only in strategy or decision criterion.

## Mechanism
A defining feature of many laboratory effects is specificity: improvement often transfers poorly to a different retinal location, orientation or eye. Karni and Sagi's texture-discrimination study is the classic demonstration, and it was taken as evidence that early visual cortex is modified. Later work showed that specificity is not universal: training protocols that are shorter or involve double training can produce substantial transfer.

Proposed mechanisms fall into two broad groups. One is changes in sensory representations, such as sharper tuning or altered response gain of neurons in early visual cortex (reviewed by Gilbert, Sigman and Crist). The other is changes in readout: the same sensory responses are reweighted so the decision stage relies on the most informative channels, and Dosher and Lu showed that behavior can be modeled as external-noise filtering plus internal-noise reduction through channel reweighting. Evidence supports both, and their relative contributions depend on task and training regime.

Attention, feedback and reward gate learning: stimuli that are task-relevant are learned, while task-irrelevant stimuli are often not. Consolidation, including sleep, contributes to stabilizing gains.

## Role in Active Inference
In the generative-model framework, perceptual learning is parameter learning: slow updates of the likelihood mapping from hidden causes to sensations, and of priors over those causes, driven by accumulated [[prediction_error]]. This gives a natural explanation of why attention and task relevance matter, since [[precision_weighting]] determines which errors drive updates, and why learning can reduce uncertainty about stimulus features. It also suggests that learning and inference occur on different timescales of the same objective.

The correspondence is a modeling choice. Standard reweighting and tuning-change accounts are formulated without free energy, and distinguishing them from a free-energy account by data alone is hard; the empirical case for any single mechanism remains open.

## Related concepts
- [[neural_plasticity]] — biological substrate of learned change
- [[learning_mechanisms]] — general learning rules
- [[perceptual_psychology]] — experimental methods for measuring thresholds
- [[feature_based_attention]] — attention to features that gates learning
- [[synaptic_plasticity]] — synaptic changes proposed to underlie it
- [[visual_perception]] — the most studied domain

## References
- Karni, A., & Sagi, D. (1991). Where practice makes perfect in texture discrimination: Evidence for primary visual cortex plasticity. Proceedings of the National Academy of Sciences.
- Gilbert, C. D., Sigman, M., & Crist, R. E. (2001). The neural basis of perceptual learning. Neuron.
- Dosher, B. A., & Lu, Z.-L. (1998). Perceptual learning reflects external noise filtering and internal noise reduction through channel reweighting. Proceedings of the National Academy of Sciences.
