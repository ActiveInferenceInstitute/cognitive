---
title: Interference
description: Interference in memory is the loss or distortion of a trace caused by competing traces, covering proactive and retroactive forms, network analogues, and links to Active Inference.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - memory
  - forgetting
  - competition
semantic_relations:
  - type: relates
    links:
      - "[[memory_systems]]"
      - "[[working_memory]]"
      - "[[stability_plasticity]]"
---

# Interference

## Overview
Interference is a leading account of forgetting in which retrieval fails because other stored material competes with or overwrites the target, not because the trace has simply decayed. The sense used here is memory interference; the related use in attention research (response conflict in Stroop and flanker tasks) is handled under [[inhibitory_control]] and [[conflict_monitoring]]. The same word is applied to interacting patterns in artificial networks, discussed below.

## Core claims
- **Retroactive interference**: learning new material impairs recall of older material (A-B list followed by A-C list reduces recall of A-B).
- **Proactive interference**: earlier material impairs learning or recall of newer material. Release from proactive interference occurs when the category of the to-be-remembered items changes.
- **Cue overload**: the more items associated with a retrieval cue, the less effective the cue is for any one of them, as in the fan effect, where response time grows with the number of facts linked to a concept.
- **Output and retrieval competition**: retrieving some items can suppress related competitors, a finding usually interpreted through inhibitory mechanisms.

Interference is largest between similar items and similar contexts. This is why distinctive encoding, context change, and temporal spacing reduce it, and why [[pattern_separation]] is proposed as a neural mechanism that decorrelates overlapping episodes.

In connectionist models, learning a new mapping can destroy an earlier one when both use the same weights, which McCloskey and Cohen described as catastrophic interference in 1989. Remedies include interleaved training, complementary fast and slow learning systems, and constraints on weight change (see [[memory_consolidation]] and [[systems_consolidation]]).

## Role in Active Inference
The connection is indirect. Active Inference does not offer a dedicated theory of forgetting, but its learning scheme faces the same trade-off. Parameters of a generative model are updated to reduce free energy on current data; if the model is too flexible and updates ignore past evidence, old regularities can be overwritten, whereas accumulating Dirichlet counts makes each new observation shift beliefs less over time. Model comparison and complexity penalties (see [[model_complexity]]) bias toward retaining parameters that earlier data supported. These are plausible modelling levers, not established explanations of human interference effects.

## Related concepts
- [[memory_systems]] — where interference sits among forms of memory
- [[working_memory]] — highly interference-limited short-term store
- [[stability_plasticity]] — the general dilemma that interference exemplifies
- [[neural_plasticity]] — substrate in which new learning can overwrite old
- [[pattern_separation]] — hippocampal mechanism that limits overlap between traces
- [[episodic_memory]] — domain in which interference is classically studied

## References
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109-165.
