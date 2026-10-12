---
title: Pruning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - development
  - plasticity
  - synapse
  - model_reduction
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[synaptic_plasticity]]"
      - "[[neural_plasticity]]"
aliases:
  - synapse_elimination
  - synaptic_pruning
wikidata: Q2124069
---

# Pruning

## Overview
Pruning is the selective elimination of synapses, axon branches, or whole connections, so that a circuit ends up with fewer, more refined connections than it began with. In neurodevelopment, synaptic pruning follows an early overproduction of synapses and continues, in the human cortex, through childhood and adolescence. The term is also used in machine learning for removing weights or units from a trained network.

## Mechanism
In humans, synaptic density in the cortex rises rapidly in early life and then declines to adult levels, with the timing differing between regions (Huttenlocher, 1979; later work on prefrontal cortex indicates a protracted decline into adolescence). The process is experience-dependent: connections that are active together are more likely to be retained, whereas weakly or asynchronously active ones are removed ("use it or lose it"). Established cellular contributors include:

- **Activity-dependent competition** among inputs for target cells, for instance in the elimination of multiple climbing-fiber inputs onto cerebellar Purkinje cells or in retinogeniculate refinement.
- **Microglia**, which engulf synaptic material; Paolicelli and colleagues (2011) showed that disrupting microglial signaling leaves excess immature synapses.
- **Complement pathway** tagging of synapses for removal.

Sleep may contribute to net synaptic downscaling (Tononi and Cirelli, 2003), although this remains debated. Abnormal pruning has been proposed to contribute to neurodevelopmental and psychiatric conditions, but causality in humans is difficult to establish.

In artificial networks, magnitude-based or gradient-based pruning removes low-importance parameters, often with little loss in accuracy; the lottery ticket hypothesis (Frankle and Carbin, 2019) reports that small trainable subnetworks exist within large ones.

## Role in Active Inference
The link is a precise one on the computational side. Bayesian model reduction scores the evidence for a reduced model, in which some parameters are removed or given tight priors, directly from a fitted full model, so redundant parameters can be eliminated while accuracy and complexity are traded off (Friston and Penny, 2011). This gives a free-energy account of why removing connections can improve a model: complexity falls without a corresponding loss of accuracy. Applied to sleep, it has been proposed as a function of offline processing, and it echoes the synaptic homeostasis proposal. Whether biological pruning implements Bayesian model reduction is a hypothesis; the match is formal and suggestive but no experiment has directly tested it.

## Related concepts
- [[structure_learning]] — Bayesian model reduction is a structure-learning operation
- [[synaptic_plasticity]] — weakening and elimination of synapses
- [[critical_periods]] — pruning peaks in sensitive windows
- [[model_complexity]] — the cost pruning reduces
- [[homeostatic_plasticity]] — complementary regulation of synaptic strength
- [[systems_consolidation]] — offline processing during sleep

## References
- Huttenlocher, P. R. (1979). Synaptic density in human frontal cortex: developmental changes and effects of aging. *Brain Research*.
- Paolicelli, R. C., et al. (2011). Synaptic pruning by microglia is necessary for normal brain development. *Science*.
- Friston, K., & Penny, W. (2011). Post hoc Bayesian model selection. *NeuroImage*.
- Frankle, J., & Carbin, M. (2019). The lottery ticket hypothesis: finding sparse, trainable neural networks. *ICLR*.
