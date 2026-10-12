---
title: Dendritic Plasticity
description: Activity-dependent changes in the structure and excitability of dendrites and dendritic spines that receive synaptic input, their mechanisms and evidence, and links to learning rules in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - synaptic_physiology
  - plasticity
  - dendrites
semantic_relations:
  - type: relates
    links:
      - "[[neural_plasticity]]"
      - "[[synaptic_plasticity]]"
      - "[[structural_plasticity]]"
      - "[[dendritic_computation]]"
      - "[[spike_timing_dependent_plasticity]]"
      - "[[hebbian_learning]]"
wikidata: Q15205473
---

# Dendritic Plasticity

## Overview
Dendrites are the branched processes through which a neuron receives most of its synaptic input. Dendritic plasticity refers to experience-dependent changes in them, including the growth, retraction and turnover of branches, the formation, enlargement and loss of dendritic spines, and modifications of the voltage-gated channels that set dendritic excitability. These changes alter which inputs a neuron integrates and how strongly.

## Mechanisms and evidence
- **Spine structural change.** Imaging of single spines showed that stimulation that induces long-term potentiation enlarges the stimulated spine (Matsuzaki et al., 2004), linking synaptic strength to spine volume. Long-term depression is associated with spine shrinkage.
- **Turnover in vivo.** Two-photon imaging in adult mouse cortex has shown that a fraction of spines appear and disappear over days to weeks, while others persist for months, and that sensory experience changes turnover rates (Trachtenberg et al., 2002; Holtmaat & Svoboda, 2009).
- **Back-propagating action potentials.** An action potential travelling from the soma into the dendrite supplies a signal that, together with synaptic input, can induce potentiation, as shown in hippocampal pyramidal neurons (Magee & Johnston, 1997).
- **Excitability changes.** Activity can also modify dendritic ion channels and local spike generation, which changes integration without changing synapse number.

Mechanisms differ across cell types, brain regions and ages, and the extent to which structural changes underlie stable memory remains under study.

## Role in Active Inference
Active Inference interprets synaptic plasticity as gradient descent on free energy, so that Hebbian-like changes accumulate evidence about the statistical structure of inputs (see [[hebbian_learning]]). Dendrites matter to the proposed biological implementation because predictions and prediction errors are thought to converge on different compartments of the same pyramidal cell (see [[deep_pyramidal_cells]]). Whether these specific compartment-level rules are used by cortex is an open research question, and the link is a hypothesis rather than an established finding.

## Related concepts
- [[neural_plasticity]] — the wider set of adaptive changes in the nervous system
- [[synaptic_plasticity]] — change in strength at individual synapses
- [[structural_plasticity]] — physical remodelling of neural structures
- [[dendritic_computation]] — computations that dendrites carry out on their inputs
- [[spike_timing_dependent_plasticity]] — a timing rule for synaptic change

## References
- Magee, J. C., & Johnston, D. (1997). A synaptically controlled, associative signal for Hebbian plasticity in hippocampal neurons. Science, 275, 209-213.
- Matsuzaki, M., Honkura, N., Ellis-Davies, G. C. R., & Kasai, H. (2004). Structural basis of long-term potentiation in single dendritic spines. Nature, 429, 761-766.
- Trachtenberg, J. T., Chen, B. E., Knott, G. W., Feng, G., Sanes, J. R., Welker, E., & Svoboda, K. (2002). Long-term in vivo imaging of experience-dependent synaptic plasticity in adult cortex. Nature, 420, 788-794.
- Holtmaat, A., & Svoboda, K. (2009). Experience-dependent structural synaptic plasticity in the mammalian brain. Nature Reviews Neuroscience, 10, 647-658.
