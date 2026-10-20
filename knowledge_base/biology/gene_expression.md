---
external_mappings:
  - target: http://purl.obolibrary.org/obo/GO_0010467
    relation: exactMatch
    evidence: https://www.ebi.ac.uk/ols4/api/ontologies/go/terms?obo_id=GO:0010467
    rationale: >-
      The source defines using gene information to make a functional product,
      protein or non-coding RNA, and covers transcription, processing and
      translation. GO defines conversion of gene sequence into a mature protein or
      RNA product, including processing and maturation. Both denote gene expression,
      rather than transcription alone.
    reviewed_on: "2026-10-20"
title: Gene Expression
description: "How genetic information is transcribed and translated into RNA and protein, how that process is regulated, and its links to development, plasticity, and regulatory-network models."
type: biology
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - molecular-biology
  - genetics
  - regulation
semantic_relations:
  - type: relates
    links:
      - "[[gene_regulatory_networks]]"
      - "[[molecular_biology]]"
      - "[[genetics]]"
wikidata: Q26972
---

# Gene Expression

## Overview
Gene expression is the process by which the information in a gene is used to make a functional product, usually a protein or a non-coding RNA. The flow of sequence information from DNA to RNA to protein, sometimes called the central dogma, was formulated by Francis Crick. How much, when and where each gene is expressed distinguishes cell types that share the same genome.

## Mechanism and regulation
In eukaryotes, transcription by RNA polymerase II produces a precursor RNA, which is capped, spliced and polyadenylated into messenger RNA. The mRNA is exported and translated by ribosomes into protein ([[protein_synthesis]]). Each step can be regulated. Transcription factors bind promoters and distant enhancers to raise or lower transcription; chromatin state, DNA methylation and histone modification make regions more or less accessible; alternative splicing yields multiple products from one gene; microRNAs and RNA-binding proteins alter mRNA stability and translation; protein degradation sets the final level.

Expression is intrinsically noisy. Elowitz and colleagues showed in single bacterial cells that genetically identical cells in an identical environment differ in protein levels, because transcription and translation involve small numbers of molecules. Expression is measured by quantitative PCR for single genes and by microarrays or RNA sequencing for genome-wide profiles, with single-cell RNA sequencing resolving differences between cells.

Interacting regulators form [[gene_regulatory_networks]], whose dynamics (feedback loops, bistable switches, oscillators) explain differentiation and pattern formation ([[morphogenesis]]). In the nervous system, neuronal activity triggers transcription of immediate-early genes, and sustained plasticity requires new gene expression and protein synthesis, the "late" phase described in [[ltp]] and [[memory_consolidation]]. Differences in expression also underlie social-insect castes arising from the same genome ([[division_of_labor]]).

## Role in Active Inference
The connection is analogical and model-based. Friston and colleagues have used free-energy formulations to model how cells in a developing tissue infer their position and regulate morphogenesis, treating cellular signalling as inference about the embryo's pattern ([[friston_2015_knowing]]). Standard gene-expression biology does not require this framing, and mapping regulatory networks onto generative models remains a research proposal rather than an established account.

## Related concepts
- [[cell_biology]] — the cellular setting for expression
- [[systems_biology]] — network-level analysis of expression data
- [[neural_development]] — expression programmes guiding brain formation
- [[structural_proteins]] — examples of expressed products supporting cell architecture

## References
- Elowitz, M. B., Levine, A. J., Siggia, E. D., & Swain, P. S. (2002). Stochastic gene expression in a single cell. *Science*, 297.
