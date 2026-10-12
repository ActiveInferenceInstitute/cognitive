---
title: Reverberatory Circuits
description: Neural loops in which activity re-excites its own source and so persists after a stimulus has ended, proposed as a basis for short-term memory and early consolidation.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neural_circuits
  - working_memory
  - persistent_activity
semantic_relations:
  - type: relates
    links:
      - "[[working_memory]]"
      - "[[hebbian_learning]]"
      - "[[attractor_dynamics]]"
      - "[[memory_consolidation]]"
      - "[[neural_circuits]]"
---

# Reverberatory Circuits

## Overview
A reverberatory circuit is a closed or recurrent chain of neurons in which activity can circulate, so that excitation continues for some time after the triggering input has stopped. The concept came from anatomical work by Lorente de Nó (1938), who described chains of interneurons that could form closed loops. Hebb (1949) used it to propose that an experience is first held as ongoing activity in cell assemblies, which later become structurally stabilised.

## Mechanism
Persistent activity requires positive feedback strong enough to offset decay but not so strong that activity runs away. Recurrent excitation, usually balanced by inhibition, can create several stable activity states. A brief input pushes the network into one of these states, which then persists until another input or inhibitory event dislodges it. This is the logic of [[attractor_dynamics]]. Computational work has emphasised slow synaptic receptors, such as NMDA receptors, as helping to stabilise reverberation, because their long time constants bridge the delay between spikes (Wang, 2001).

## Evidence and limits
Delay-period firing that outlasts a stimulus has been recorded in prefrontal cortex and other regions of animals performing working-memory tasks. This is consistent with reverberation, but it does not prove it, since persistent firing could also arise from intrinsic cellular properties, long-lasting synaptic changes or activity-silent traces. Population recordings have also shown that delay activity is often dynamic rather than constant. The idea that reverberation alone supports consolidation has been replaced by richer accounts involving synaptic tagging and sleep-dependent replay.

## Relation to Active Inference
In generative-model terms, sustained activity can be read as maintenance of a belief about a hidden state over time, with recurrent dynamics acting as the transition model. Hebb's cell assemblies and attractor networks fit this description, but this is an analogy; Active Inference does not commit to reverberation as the biological implementation of working memory.

## Related concepts
- [[working_memory]] — the function most often linked to persistent activity
- [[hebbian_learning]] — the learning rule that Hebb paired with reverberation
- [[attractor_dynamics]] — the dynamical systems description of stable states
- [[memory_consolidation]] — the process reverberation was first proposed to start
- [[synaptic_tagging]] — a later account of how traces are stabilised

## References
- Lorente de Nó, R. (1938). Analysis of the activity of the chains of internuncial neurons. Journal of Neurophysiology, 1, 207-244.
- Hebb, D. O. (1949). The Organization of Behavior: A Neuropsychological Theory. Wiley.
- Wang, X.-J. (2001). Synaptic reverberation underlying mnemonic persistent activity. Trends in Neurosciences, 24, 455-463.
