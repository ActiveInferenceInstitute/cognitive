---
title: CA3
description: The CA3 field of the hippocampus, its recurrent collateral network and mossy fibre input, theories of autoassociative memory and pattern completion, and links to attractor models.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - hippocampus
  - memory
  - attractor_networks
semantic_relations:
  - type: relates
    links:
      - "[[hippocampus]]"
      - "[[ca1]]"
      - "[[dentate_gyrus]]"
      - "[[pattern_completion]]"
      - "[[hopfield_networks]]"
---

# CA3

## Overview
CA3 is the hippocampal subfield that lies between the [[dentate_gyrus]] and [[ca1]]. It is distinguished by dense recurrent connections among its own pyramidal cells, the recurrent collaterals, in addition to the inputs it receives from the dentate gyrus and the entorhinal cortex. This architecture, together with its role in the trisynaptic circuit, has made it a main candidate for the storage and retrieval of episodic memories by an autoassociative network.

## Anatomy
- **Mossy fibres.** Axons of dentate granule cells form large, sparse synapses on CA3 pyramidal cells. Each CA3 cell receives relatively few of them, but they are strong, and they have been proposed to act as teaching inputs when new patterns are stored.
- **Recurrent collaterals.** A large share of the excitatory input to a CA3 pyramidal cell comes from other CA3 cells, which makes the area unusual among cortical regions in its degree of recurrence.
- **Perforant path.** Direct inputs from layer II of the entorhinal cortex carry a cue that can drive recall.
- **Output.** CA3 projects to CA1 through the Schaffer collaterals (see [[entorhinal_cortex]]).

## Theory and evidence
Marr (1971) proposed that a recurrently connected archicortex could store patterns by modifiable synapses and reconstruct a whole pattern from a part. This idea was developed into models in which Hebbian synaptic modification stores patterns as attractors, and a partial cue leads the network to complete the stored pattern (Treves and Rolls, 1994; see [[hebbian_learning]] and [[hopfield_networks]]). In the standard account, the dentate gyrus supports [[pattern_separation]] on storage while CA3 supports [[pattern_completion]] on retrieval. Supporting evidence includes a mouse study in which selective loss of NMDA receptors in CA3 impaired recall of a spatial memory from partial cues without preventing learning from full cues (Nakazawa et al., 2002). Recording studies are consistent with attractor-like dynamics, but they also show more graded behaviour than a simple model predicts.

## Relation to Active Inference
The link combines formal parallels with interpretation. Attractor networks of the Hopfield type can be written as energy-based models, and retrieval from a partial cue can be described as inference in a generative model of stored patterns, which links CA3 to the idea of completing missing observations from a learned model (see [[energy_based_models]] and [[generative_model]]). Applying these ideas to CA3 in Active Inference terms remains a theoretical proposal.

## Related concepts
- [[hippocampus]] — the larger system
- [[pattern_completion]] — reconstruction of a stored pattern from a cue
- [[pattern_separation]] — making similar inputs distinct
- [[hopfield_networks]] — attractor models of associative memory
- [[episodic_memory]] — memory type that CA3 is thought to support

## References
- Marr, D. (1971). Simple memory: A theory for archicortex. Philosophical Transactions of the Royal Society of London B, 262, 23-81.
- Treves, A., & Rolls, E. T. (1994). Computational analysis of the role of the hippocampus in memory. Hippocampus, 4, 374-391.
- Nakazawa, K., Quirk, M. C., Chitwood, R. A., et al. (2002). Requirement for hippocampal CA3 NMDA receptors in associative memory recall. Science, 297, 211-218.
