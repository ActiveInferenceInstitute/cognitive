---
title: Late LTP
description: The long-lasting, protein-synthesis-dependent phase of long-term potentiation that stabilises synaptic strengthening for hours or longer, and its place in models of memory consolidation.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - synaptic_plasticity
  - memory_consolidation
  - molecular_neuroscience
semantic_relations:
  - type: relates
    links:
      - "[[ltp]]"
      - "[[early_ltp]]"
      - "[[synaptic_tagging]]"
      - "[[memory_consolidation]]"
      - "[[neural_plasticity]]"
---

# Late LTP

## Overview
Late long-term potentiation (L-LTP) is the persistent phase of [[ltp]], the activity-dependent strengthening of synapses first described by Bliss and Lomo in the rabbit dentate gyrus. Unlike [[early_ltp]], which decays within one to a few hours, L-LTP lasts for many hours and, in suitable preparations and in vivo, for days or longer. It is distinguished by a requirement for new gene expression and protein synthesis.

## Induction and mechanism
L-LTP in hippocampal area CA1 is classically induced by repeated, spaced bouts of high-frequency stimulation, which produce stronger and longer calcium signalling through [[nmda_receptors]] than a single bout. Downstream signalling involves protein kinase A, MAP kinase and the transcription factor CREB, leading to transcription of plasticity-related genes and local protein synthesis near activated synapses (see [[protein_synthesis]]). Products of these processes support growth and remodelling of dendritic spines and an increase in postsynaptic receptor content. Pharmacological block of translation or transcription during induction leaves early LTP intact but abolishes the late phase, which is the operational criterion for the distinction.

## Synaptic tagging and capture
Because proteins are made in the cell body or dendrites but must act at specific synapses, input specificity requires a marker. The synaptic tagging and capture hypothesis proposes that weakly stimulated synapses are tagged and can capture plasticity-related proteins produced following strong stimulation of other inputs (see [[synaptic_tagging]]). Frey and Morris reported experimental support in 1997.

## Relevance to memory
L-LTP is a leading cellular candidate for the stabilisation of memories during consolidation, since protein-synthesis inhibitors given after learning impair long-term but not short-term retention in many tasks (see [[memory_consolidation]] and [[hippocampus]]). Whether LTP is the physical substrate of memory remains debated, and L-LTP is one mechanism among several.

## Relation to Active Inference
The link is an analogy. Learning of likelihood and transition parameters in discrete Active Inference models is often written as accumulating Dirichlet counts, which resembles activity-dependent strengthening, and consolidation has been compared to offline model optimisation. No mapping from L-LTP biophysics to a specific update rule is established.

## Related concepts
- [[ltp]] — the parent phenomenon
- [[early_ltp]] — the short-lived phase independent of protein synthesis
- [[synaptic_tagging]] — how late-phase proteins reach specific synapses
- [[memory_consolidation]] — the memory process L-LTP is thought to serve
- [[hippocampus]] — the structure where L-LTP is best characterised

## References
- Bliss, T. V. P., & Lomo, T. (1973). Long-lasting potentiation of synaptic transmission in the dentate area of the anaesthetized rabbit following stimulation of the perforant path. Journal of Physiology, 232, 331-356.
- Frey, U., & Morris, R. G. M. (1997). Synaptic tagging and long-term potentiation. Nature, 385, 533-536.
- Kandel, E. R. (2001). The molecular biology of memory storage: A dialogue between genes and synapses. Science, 294, 1030-1038.
