---
title: Long-Term Potentiation
description: "Long-term potentiation as an activity-dependent, lasting strengthening of synapses, its NMDA receptor mechanism, early and late phases, and its relation to Hebbian learning and memory."
type: biology
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - synaptic-plasticity
  - memory
  - hippocampus
semantic_relations:
  - type: relates
    links:
      - "[[synaptic_plasticity]]"
      - "[[hebbian_learning]]"
      - "[[memory_consolidation]]"
---

# Long-Term Potentiation

## Overview
Long-term potentiation (LTP) is a persistent increase in synaptic strength following particular patterns of activity, lasting from tens of minutes to days or longer in vivo. It was first reported by Bliss and Lomo in 1973 in the rabbit dentate gyrus after brief high-frequency stimulation of the perforant path, and it is the best-studied cellular model of [[synaptic_plasticity]] and a leading candidate substrate for learning.

## Mechanism
The canonical form at hippocampal CA3-CA1 synapses depends on the NMDA receptor, which acts as a coincidence detector: it opens only when glutamate is bound and the postsynaptic membrane is sufficiently depolarized to remove the magnesium block. The resulting calcium entry into the spine activates kinases, notably CaMKII. Early-phase LTP (roughly the first one to three hours) is expressed by phosphorylation of AMPA receptors and insertion of additional AMPA receptors into the postsynaptic membrane, raising the response to a given release of glutamate. Late-phase LTP needs transcription and new protein synthesis, engaging signalling through CREB and altered [[gene_expression]], and is accompanied by growth of the spine, which links it to [[structural_plasticity]] and [[structural_proteins]].

Induction depends on timing and rate. High-frequency tetanus, theta-burst stimulation and spike-timing protocols (presynaptic spike shortly before postsynaptic spike) all induce LTP, while low-frequency stimulation or the reverse timing tends to induce long-term depression, its counterpart. The threshold for LTP shifts with prior activity, a property studied as [[metaplasticity]].

Evidence tying LTP to behavioral memory is substantial but largely correlational and pharmacological: blocking NMDA receptors in the hippocampus impairs certain spatial learning tasks in rodents, and learning-related potentiation has been observed in some circuits, including the amygdala in fear conditioning ([[amygdala]]). Whether LTP is the memory trace itself or one contributing mechanism is debated.

## Role in Active Inference
LTP provides a biological implementation of the correlation-based rule in [[hebbian_learning]]. In neural-process theories of Active Inference and predictive coding, learning of connection strengths (the parameters of the generative model) is proposed to be a Hebbian-like update proportional to presynaptic activity times prediction error ([[predictive_coding]]). The correspondence is a theoretical proposal about plausibility, not a demonstration that LTP implements variational free-energy gradients.

## Related concepts
- [[synaptic_consolidation]] — stabilization of potentiated synapses
- [[memory_systems]] — systems-level view of where plastic changes matter
- [[neural_plasticity]] — broader class of experience-dependent change
- [[homeostatic_plasticity]] — mechanism that keeps potentiation within bounds

## References
- Bliss, T. V. P., & Lomo, T. (1973). Long-lasting potentiation of synaptic transmission in the dentate area of the anaesthetized rabbit following stimulation of the perforant path. *Journal of Physiology*, 232.
