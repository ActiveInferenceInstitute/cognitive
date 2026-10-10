---
title: Pattern Separation
description: Pattern separation is the transformation of similar inputs into less overlapping stored representations, attributed to the dentate gyrus, and this page covers its computation and relation to memory and Active Inference.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - hippocampus
  - memory
  - neural-coding
semantic_relations:
  - type: relates
    links:
      - "[[pattern_completion]]"
      - "[[episodic_memory]]"
      - "[[memory_organization]]"
---

# Pattern Separation

## Overview
Pattern separation is the process by which a neural system makes the representations of similar inputs more distinct, so that overlapping experiences can be stored without being confused. It is paired with pattern completion, the recovery of a full stored pattern from a partial or noisy cue ([[pattern_completion]]). The two are central to computational theories of hippocampal memory (Marr, 1971; McNaughton and Morris, 1987; Treves and Rolls, 1994).

## Mechanism
The standard proposal assigns pattern separation to the dentate gyrus (DG) and pattern completion to area CA3:
- **Expansion and sparsity**: entorhinal input is projected onto a much larger population of dentate granule cells, of which only a small fraction is active at any time. Random expansion followed by sparse thresholding reduces the correlation between the output patterns relative to the inputs.
- **Sparse mossy fibre input**: each granule cell contacts few CA3 pyramidal cells with high-efficacy synapses, which can impose a nearly arbitrary code on CA3 during encoding.
- **Adult neurogenesis**: newly born granule cells, with high excitability, are proposed to support discrimination of similar contexts; evidence is stronger in rodents than in humans.
- **Behavioural assays**: in rodents, discrimination of similar spatial locations or contexts depends on DG. In humans, tasks that ask whether an image is identical to or merely similar to a studied image (mnemonic similarity tasks) are interpreted as behavioural proxies, with high-resolution fMRI signals in DG and CA3 correlated with lure discrimination (Yassa and Stark, 2011).

Capacity limits and interference between similar memories motivate the mechanism ([[interference]]), and impaired separation with age has been proposed to contribute to memory confusions, a claim that remains under study.

## Role in Active Inference
In generative-model terms, separation corresponds to learning distinct hidden-state representations for observations that are similar but predict different outcomes, so state inference can disambiguate them, whereas completion corresponds to inferring the latent cause behind partial observations. Hippocampal accounts within the free-energy tradition treat the hippocampus as inferring latent states of a sequence or scene model, with complexity penalties favouring compact but discriminative codes. The connection is a modelling interpretation; the original separation and completion theories are written in terms of network associative memory, and a derivation of the DG-CA3 division from free-energy minimization has not been established.

## Related concepts
- [[pattern_completion]] — the complementary retrieval operation
- [[episodic_memory]] — memory type that depends on separation
- [[memory_organization]] — architecture in which these processes operate
- [[interference]] — confusion that separation reduces
- [[sparse_coding]] — coding principle underlying the DG expansion
- [[neural_computation]] — computational view of circuit operations

## References
- McNaughton, B. L., & Morris, R. G. M. (1987). Hippocampal synaptic enhancement and information storage within a distributed memory system. *Trends in Neurosciences*, 10, 408-415.
- Treves, A., & Rolls, E. T. (1994). Computational analysis of the role of the hippocampus in memory. *Hippocampus*, 4, 374-391.
- Marr, D. (1971). Simple memory: A theory for archicortex. *Philosophical Transactions of the Royal Society B*, 262, 23-81.
- Yassa, M. A., & Stark, C. E. L. (2011). Pattern separation in the hippocampus. *Trends in Neurosciences*, 34, 515-525.
