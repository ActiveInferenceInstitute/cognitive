---
title: Gain Control
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - neural-computation
  - normalization
  - attention
semantic_relations:
  - type: relates
    links:
      - "[[precision_weighting]]"
      - "[[attention_mechanisms]]"
      - "[[neural_computation]]"
---

# Gain Control

## Overview
Gain control is the adjustment of how strongly a neuron or population responds to a given input. Gain is the slope of the input-output relation; changing it rescales responses without necessarily changing what they are tuned to. It operates over time scales from milliseconds (contrast normalization) to minutes (adaptation) and is modulated by attention, arousal and behavioral state.

## Mechanism
The best-characterized form is divisive normalization. A neuron's driving input is divided by a pooled measure of activity in a surrounding population:

$R_i = \gamma \, \frac{D_i^{\,n}}{\sigma^{n} + \sum_j w_{ij} D_j^{\,n}}$

where $D_i$ is the neuron's driving input, $\sigma$ a semisaturation constant, and the sum runs over a normalization pool. Heeger formalized this to explain contrast saturation and cross-orientation suppression in cat visual cortex, and Carandini and Heeger review it as a canonical computation found across sensory modalities and species.

Normalization produces contrast gain control: response curves shift so the system stays sensitive across orders of magnitude of input intensity, which compresses a wide dynamic range into the limited firing range of neurons. Adaptation to a prevailing stimulus level has similar effects. Gain can also be set by neuromodulators and by top-down signals: attention increases responses to attended stimuli, an effect captured by the normalization model of attention in which attention modifies the input drive before the normalization stage (Reynolds and Heeger). Other mechanisms include shunting inhibition, synaptic depression, and changes in network state.

Multiplicative gain fields, in which sensory responses are scaled by eye or head position in parietal cortex, allow coordinate transformations to be computed in populations.

## Role in Active Inference
Feldman and Friston proposed that the gain of prediction-error units encodes the expected precision of the corresponding signal, so synaptic or neuromodulatory gain control is how [[precision_weighting]] could be realized neurally. Attention, arousal and neuromodulation (for example acetylcholine and dopamine) are then discussed as precision-setting mechanisms.

The identification is not exact. Divisive normalization is a specific nonlinear computation with its own functional rationale (efficient use of dynamic range), and it is not the same thing as a scalar precision parameter. Some work argues that normalization can be derived from inference about contrast or reliability, but that claim is model-dependent.

## Related concepts
- [[precision_weighting]] — the active-inference construct gain is proposed to implement
- [[visual_attention]] — attentional modulation of gain
- [[attention_mechanisms]] — broader attentional accounts
- [[population_coding]] — normalization acts across a population
- [[sensory_processing]] — contexts where adaptation and normalization are observed
- [[neural_computation]] — normalization as a canonical operation

## References
- Heeger, D. J. (1992). Normalization of cell responses in cat striate cortex. Visual Neuroscience.
- Carandini, M., & Heeger, D. J. (2012). Normalization as a canonical neural computation. Nature Reviews Neuroscience.
- Reynolds, J. H., & Heeger, D. J. (2009). The normalization model of attention. Neuron.
- Feldman, H., & Friston, K. J. (2010). Attention, uncertainty, and free-energy. Frontiers in Human Neuroscience.
