---
title: Heterosynaptic Plasticity
description: Activity-dependent change at synapses that were not themselves active, its forms, its role in stabilising Hebbian learning, and how it relates to modulatory and prediction-based learning rules.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - synaptic_plasticity
  - learning
  - neuromodulation
semantic_relations:
  - type: relates
    links:
      - "[[homosynaptic]]"
      - "[[synaptic_plasticity]]"
      - "[[hebbian_learning]]"
      - "[[synaptic_tagging]]"
      - "[[homeostatic_plasticity]]"
---

# Heterosynaptic Plasticity

## Overview
Heterosynaptic plasticity is a lasting change in the strength of a synapse that is caused by activity at other synapses, and not by that synapse's own activity. It contrasts with [[homosynaptic]] plasticity, in which change is confined to the inputs that were activated. The term covers several distinct phenomena, which differ in mechanism and in whether the change is an increase or a decrease in efficacy.

## Forms
- **Modulatory facilitation.** In the sea snail Aplysia, a noxious stimulus to the tail activates facilitatory interneurons that release serotonin onto sensory neuron terminals, enhancing transmission at synapses that were not active during the stimulus. This underlies sensitisation of the withdrawal reflex and was studied extensively by Kandel and colleagues.
- **Compensatory change.** Strong potentiation at some synapses on a neuron is accompanied by weakening of neighbouring inactive synapses. This is often described as a means of conserving the total synaptic weight of a neuron.
- **Spread of plasticity-related products.** Proteins made after strong induction can be captured by recently active synapses, so that a weak input becomes stabilised, as in synaptic tagging (see [[synaptic_tagging]]).

## Functional significance
Hebbian potentiation alone is unstable, since stronger synapses promote further activity and so further strengthening. Bailey and colleagues (2000) discussed heterosynaptic modulation as a process that can stabilise Hebbian plasticity and memory, and Chistiakova and colleagues (2015) reviewed experiments and models in which heterosynaptic changes act as a homeostatic counterweight, preventing runaway growth and preserving the differences between synapses established by learning. Modulators such as dopamine, noradrenaline and acetylcholine gate which synapses are modifiable, and so supply a heterosynaptic route for task-relevant signals to shape learning (see [[neuromodulation]]).

## Relation to Active Inference
The link is indirect. Active Inference process theories require learning rules in which synaptic updates depend on pre-synaptic activity, post-synaptic activity and modulatory signals such as precision or prediction errors, which resembles three-factor rules with a heterosynaptic modulatory term. This is a conceptual mapping, and the specific biophysical mechanisms above are not derived from the theory.

## Related concepts
- [[homosynaptic]] — input-specific plasticity
- [[synaptic_plasticity]] — the general phenomenon of synaptic change
- [[hebbian_learning]] — the associative rule that heterosynaptic effects stabilise
- [[synaptic_tagging]] — capture of shared proteins by active synapses
- [[homeostatic_plasticity]] — regulation of overall activity or weight

## References
- Bailey, C. H., Giustetto, M., Huang, Y.-Y., Hawkins, R. D., & Kandel, E. R. (2000). Is heterosynaptic modulation essential for stabilizing Hebbian plasticity and memory? Nature Reviews Neuroscience, 1, 11-20.
- Chistiakova, M., Bannon, N. M., Chen, J.-Y., Bazhenov, M., & Volgushev, M. (2015). Homeostatic role of heterosynaptic plasticity: Models and experiments. Frontiers in Computational Neuroscience, 9, 89.
