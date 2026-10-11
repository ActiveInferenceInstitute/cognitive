---
title: Synaptic Tagging
description: The synaptic tagging and capture hypothesis, in which a transient synaptic tag allows weakly stimulated synapses to capture plasticity-related proteins and so convert short-lasting changes into persistent ones.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - synaptic_plasticity
  - memory_consolidation
  - long_term_potentiation
semantic_relations:
  - type: relates
    links:
      - "[[ltp]]"
      - "[[synaptic_plasticity]]"
      - "[[synaptic_consolidation]]"
      - "[[memory_formation]]"
      - "[[hippocampus]]"
---

# Synaptic Tagging

## Overview
Synaptic tagging, more fully the synaptic tagging and capture hypothesis, addresses a problem in memory formation: lasting synaptic changes depend on newly made proteins produced in the cell body, yet the changes themselves are specific to individual synapses. The hypothesis proposes that stimulation sets a short-lived, synapse-specific tag, and that tagged synapses capture the plasticity-related proteins made by the neuron, so that only tagged synapses are stabilised. It is the standard framework for understanding how synapse specificity and slow protein synthesis are reconciled.

## Experimental basis
- **Early and late phases.** In hippocampal slices, a weak tetanus induces a short-lasting form of long-term potentiation (early LTP) that decays within hours, whereas a strong tetanus induces a late form that depends on protein synthesis (see [[ltp]]).
- **Rescue of weak inputs.** Frey and Morris (1997) found that a weak input, which on its own produced only early LTP, was converted into persistent LTP if a strong tetanus was delivered to a separate input onto the same population of neurons within a limited time window. This suggested that the weakly stimulated synapses were tagged and could capture proteins triggered by the strong stimulation.
- **Reverse order.** The effect also works when the strong stimulation comes first, within a limited time window (of the order of an hour or more), which argues that the proteins persist and the tag is the limiting, transient element.
- **Molecular candidates.** The tag is thought to involve local changes such as kinase activity and cytoskeletal modification, but its precise molecular identity remains incompletely resolved (Redondo & Morris, 2011).

The hypothesis has been extended to behaviour, where weak learning events close in time to novel or salient experiences can be strengthened, which suggests a way in which unrelated memories become linked.

## Relation to Active Inference
The connection is weak and indirect. Active Inference does not model protein synthesis, but synaptic tagging illustrates a general requirement of learning rules: a local eligibility trace that marks which synapses are candidates for change, with a later global or neuromodulatory signal deciding whether to consolidate. Some Active Inference and predictive coding treatments describe learning as precision-gated or neuromodulated in a comparable way, which is a loose analogy rather than a derivation.

## Related concepts
- [[ltp]] — the form of plasticity whose persistence tagging explains
- [[synaptic_plasticity]] — general activity-dependent change in synaptic strength
- [[synaptic_consolidation]] — stabilisation of synaptic change over hours
- [[memory_formation]] — the behavioural process these changes support
- [[hippocampus]] — the structure where tagging was characterised
- [[memory_consolidation]] — broader stabilisation of memory traces

## References
- Frey, U., & Morris, R. G. M. (1997). Synaptic tagging and long-term potentiation. Nature, 385, 533-536.
- Redondo, R. L., & Morris, R. G. M. (2011). Making memories last: The synaptic tagging and capture hypothesis. Nature Reviews Neuroscience, 12, 17-30.
