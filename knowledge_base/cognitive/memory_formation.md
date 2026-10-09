---
title: Memory Formation
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - memory
  - encoding
  - hippocampus
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[memory_consolidation]]"
      - "[[memory_systems]]"
---

# Memory Formation

## Overview
Memory formation is the sequence of processes by which an experience becomes a lasting, retrievable trace: encoding, short-term maintenance, consolidation, and storage, with retrieval as the test of whether it succeeded. The term covers both the cellular changes that store information and the cognitive operations (attention, elaboration) that determine what gets stored.

## Mechanism
- **Encoding**: perceptual and conceptual features are bound into a representation. Attention and depth of processing strongly influence later recall; deeper, meaning-based processing generally yields better retention than superficial processing. In the medial temporal lobe, the hippocampus binds features into an episode, while surrounding cortices represent item and context information.
- **Maintenance**: information is held in working memory, which is limited in capacity and duration (see [[working_memory]]).
- **Consolidation**: initial changes are stabilized, first at synapses (see [[synaptic_consolidation]]) and then through reorganization across regions (see [[systems_consolidation]]).
- **Storage and retrieval**: stored traces are reactivated from partial cues, and retrieval itself can modify the trace.

At the cellular level, activity-dependent synaptic change such as LTP and LTD is the leading candidate substrate. Different memory types depend on different systems: declarative memory on the medial temporal lobe, procedural and habit learning on the basal ganglia and cerebellum, and conditioned emotional memory on the amygdala (see [[memory_systems]]). Forgetting is part of the picture, resulting from decay, interference, and retrieval failure, and is not simply a defect.

## Role in Active Inference
In active inference, memory is not stored items but parameters and states of a generative model. Formation corresponds to updating those parameters from experience (see [[hebbian_learning]]), and recall to inferring the latent states that best explain a partial cue, which is a form of pattern completion. Hippocampal function has been modeled as inference over scene or episode representations, linking spatial and episodic memory. The framework predicts that surprising or poorly predicted events carry more information to learn from, which is consistent with, but does not by itself explain, the observed memory advantage for novelty and prediction-error events. These accounts are well-posed modeling proposals; the mapping of specific model quantities to hippocampal and cortical circuitry is hypothetical, and encoding effects such as depth of processing are not derived from the formalism.

## Related concepts
- [[memory_consolidation]] — stabilization after encoding
- [[working_memory]] — maintenance stage
- [[episodic_memory]] — memory for specific events
- [[synaptic_plasticity]] — cellular mechanisms of storage
- [[neural_plasticity]] — broader capacity for structural and functional change
- [[learning_mechanisms]] — related processes that produce stored change

## References
- Squire, L. R. (1992). Memory and the hippocampus: a synthesis from findings with rats, monkeys, and humans. *Psychological Review*.
