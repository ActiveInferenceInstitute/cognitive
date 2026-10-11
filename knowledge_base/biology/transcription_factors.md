---
title: Transcription Factors
description: Sequence-specific DNA-binding proteins that increase or decrease the transcription of target genes, including their structural classes, combinatorial control, and role in gene regulatory networks and development.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - gene_regulation
  - molecular_biology
  - development
semantic_relations:
  - type: relates
    links:
      - "[[gene_expression]]"
      - "[[gene_regulatory_networks]]"
      - "[[molecular_biology]]"
      - "[[morphogenesis]]"
      - "[[cell_biology]]"
---

# Transcription Factors

## Overview
Transcription factors are proteins that bind specific DNA sequences, usually in promoters or distal enhancers, and influence whether RNA polymerase transcribes nearby genes. Together with chromatin state and noncoding regulatory DNA, they determine which genes are active in a given cell at a given time (see [[gene_expression]]). The human genome encodes roughly 1,600 likely transcription factors (Lambert et al., 2018 catalogued over 1,600), a substantial fraction of all protein-coding genes.

## Structure and mechanism
- **DNA-binding domains.** Families are classified by domain type, including zinc fingers, homeodomains, basic helix-loop-helix, basic leucine zipper and nuclear receptor domains. Each domain recognises short sequence motifs, typically 6 to 12 base pairs, with some tolerance for variation.
- **Effector domains.** Activators recruit co-activators, chromatin remodellers and the general transcription machinery, while repressors recruit factors that compact chromatin or block assembly.
- **Regulation of the factors themselves.** Activity can be controlled by ligand binding (as for steroid receptors), phosphorylation, nuclear localisation, protein partner availability and degradation.
- **Combinatorial control.** A given gene is usually regulated by several transcription factors acting together at an enhancer, so a modest number of factors can produce many expression patterns.

## Historical and conceptual role
Jacob and Monod's work on the lac operon in bacteria established the idea of a repressor that responds to a small molecule and switches genes on or off. In eukaryotes, transcription factors are central to cell identity: forced expression of a small set of them can reprogram one cell type into another, as in the generation of induced pluripotent stem cells. In development, graded and interacting factors organise body patterning.

## Relation to Active Inference
Transcription factors form the nodes and edges of [[gene_regulatory_networks]], whose dynamics can be modelled as dynamical systems or Bayesian networks. Some theoretical work in the free energy literature treats cell differentiation and pattern formation as processes in which cells infer their position or fate from signals, an active research direction in [[morphogenesis]]. These accounts are modelling hypotheses. Molecular biology itself explains transcription factor action through binding and kinetics and does not require an inference interpretation.

## Related concepts
- [[gene_expression]] — the process that transcription factors regulate
- [[gene_regulatory_networks]] — networks formed by regulatory interactions among factors and targets
- [[molecular_biology]] — the field that established the mechanisms
- [[morphogenesis]] — developmental patterning controlled by regulatory proteins
- [[cell_biology]] — cell identity and signalling context

## References
- Jacob, F., & Monod, J. (1961). Genetic regulatory mechanisms in the synthesis of proteins. Journal of Molecular Biology, 3, 318-356.
- Ptashne, M., & Gann, A. (2002). Genes and Signals. Cold Spring Harbor, NY: Cold Spring Harbor Laboratory Press.
- Lambert, S. A., et al. (2018). The human transcription factors. Cell, 172, 650-665.
