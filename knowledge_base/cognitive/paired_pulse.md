---
title: Paired Pulse
description: The paired-pulse protocol, in which two stimuli are delivered at a short interval and the second response is compared with the first, used to study short-term synaptic facilitation and depression.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - synaptic_physiology
  - short_term_plasticity
  - electrophysiology
semantic_relations:
  - type: relates
    links:
      - "[[synaptic_plasticity]]"
      - "[[neural_plasticity]]"
      - "[[ltp]]"
      - "[[homosynaptic]]"
      - "[[interneurons]]"
---

# Paired Pulse

## Overview
Paired-pulse stimulation is an experimental protocol in which two identical stimuli are delivered to a pathway separated by a short interval, and the size of the second response is compared with that of the first. The ratio of the second to the first response, the paired-pulse ratio, changes with the interval and with the properties of the synapse. A ratio above one indicates paired-pulse facilitation and a ratio below one indicates paired-pulse depression. The protocol is used in brain slices, in vivo recordings and, with magnetic stimulation, in human cortex.

## Mechanisms
At many excitatory synapses, a second stimulus arriving tens to a few hundred milliseconds after the first elicits a larger postsynaptic response. The standard explanation is presynaptic. Calcium that entered the terminal during the first spike has not yet been cleared, and residual calcium adds to the entry caused by the second spike, so the probability of transmitter release is higher (Katz and Miledi, 1968; Zucker and Regehr, 2002). Synapses with low initial release probability tend to facilitate, whereas those with high release probability tend to show depression, because the pool of releasable vesicles is partly depleted.

At very short intervals in some circuits, inhibition is recruited after the first stimulus and suppresses the second response. Because the effect depends largely on presynaptic factors, a change in the paired-pulse ratio after an experimental manipulation is commonly taken as evidence that the manipulation altered release probability, although this inference needs confirming with other measures.

## Role in plasticity research
Paired-pulse measures are used alongside long-term potentiation ([[ltp]]) to determine whether changes in synaptic strength have a presynaptic or postsynaptic locus. They also serve as a short-term, transient form of plasticity that acts on a timescale of milliseconds to seconds, which is distinct from the persistent changes discussed in [[synaptic_plasticity]] and [[neural_plasticity]]. In human work, paired-pulse transcranial magnetic stimulation is used to probe cortical excitability and inhibition.

## Relation to Active Inference
The relation is indirect. Short-term synaptic dynamics shape how the effective gain of a connection varies with recent input, and biophysical models used in neural mass modelling include such dynamics. Active Inference does not itself predict paired-pulse facilitation, and any identification of synaptic gain with the precision of prediction errors is an interpretive step rather than an established result.

## Related concepts
- [[synaptic_plasticity]] — activity-dependent changes in synaptic strength
- [[ltp]] — a long-lasting increase in synaptic efficacy
- [[homosynaptic]] — plasticity restricted to the stimulated synapse
- [[interneurons]] — inhibitory cells recruited at short intervals
- [[neural_plasticity]] — the broader family of adaptive changes in neural circuits

## References
- Katz, B., & Miledi, R. (1968). The role of calcium in neuromuscular facilitation. Journal of Physiology, 195, 481-492.
- Zucker, R. S., & Regehr, W. G. (2002). Short-term synaptic plasticity. Annual Review of Physiology, 64, 355-405.
