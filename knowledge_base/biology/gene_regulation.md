---
title: Gene Regulation
description: The mechanisms that control when and how much of each gene product is made, from transcription and chromatin to translation, and their treatment as networks and in free-energy accounts of cells.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - molecular_biology
  - transcription
  - regulation
semantic_relations:
  - type: relates
    links:
      - "[[gene_expression]]"
      - "[[gene_regulatory_networks]]"
      - "[[transcription_factors]]"
      - "[[epigenetics]]"
      - "[[molecular_biology]]"
      - "[[protein_synthesis]]"
---

# Gene Regulation

## Overview
Gene regulation is the set of processes that determine whether, when and to what extent a gene is expressed. Every cell of an organism carries essentially the same genome, yet cells differ in type and respond to conditions because they express different subsets of genes. Regulation operates at many steps between DNA and the functional product.

## Mechanisms
- **Transcriptional control.** Transcription factors bind promoters and distal enhancers, recruit or block RNA polymerase, and combine in ways that implement logic over multiple inputs (see [[transcription_factors]]).
- **Chromatin state.** DNA packaging, histone modifications and DNA methylation make regions accessible or silent, and some marks can be maintained through cell division (see [[epigenetics]]).
- **RNA processing and stability.** Alternative splicing, RNA modification, microRNAs and decay rates determine which transcripts persist.
- **Translation and protein control.** Initiation of translation, protein modification and degradation set the final level of an active protein (see [[protein_synthesis]]).

The classic model is the bacterial lac operon. Jacob and Monod (1961) proposed that a repressor protein binds an operator site to block transcription, and that an inducer, a lactose derivative, releases it, so that the genes are expressed only when the sugar is present. Eukaryotic regulation involves many more components, but feedback and combinatorial control remain central.

## Networks and noise
Interacting regulators form gene regulatory networks, whose motifs such as feedback loops can give switches, oscillators and memory (see [[gene_regulatory_networks]]). Expression within single cells is also stochastic, because few molecules are involved, and identical cells in identical conditions can differ in protein level (Elowitz et al., 2002).

## Relation to Active Inference
The link is a developing research programme with a strong theoretical component. Cells can be treated as systems that maintain their organisation by minimising free energy, with regulatory networks encoding a model of the environment and morphogenesis described as pattern regulation (Friston et al., 2015). These models are generally abstract simulations rather than reconstructions of specific molecular circuits, and their empirical support at the molecular level is limited.

## Related concepts
- [[gene_expression]] — the process of reading out genes
- [[gene_regulatory_networks]] — network descriptions of regulation
- [[transcription_factors]] — proteins that control transcription
- [[epigenetics]] — heritable regulation not changing the DNA sequence
- [[cellular_intelligence]] — information processing in single cells

## References
- Jacob, F., & Monod, J. (1961). Genetic regulatory mechanisms in the synthesis of proteins. Journal of Molecular Biology, 3, 318-356.
- Elowitz, M. B., Levine, A. J., Siggia, E. D., & Swain, P. S. (2002). Stochastic gene expression in a single cell. Science, 297, 1183-1186.
- Friston, K., Levin, M., Sengupta, B., & Pezzulo, G. (2015). Knowing one's place: a free-energy approach to pattern regulation. Journal of the Royal Society Interface, 12, 20141383.
