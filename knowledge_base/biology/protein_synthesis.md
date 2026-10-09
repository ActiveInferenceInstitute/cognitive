---
title: Protein Synthesis
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - protein_synthesis
  - translation
  - gene_expression
  - molecular_biology
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/biology/molecular_biology]]"
      - "[[knowledge_base/biology/biochemistry]]"
      - "[[knowledge_base/biology/gene_regulatory_networks]]"
---

# Protein Synthesis

## Overview
Protein synthesis is the cellular process that converts the sequence information in DNA into polypeptide chains. It has two main stages, transcription of a gene into messenger RNA and translation of that mRNA into protein by the ribosome, a flow of information summarized by Crick's central dogma.

## Mechanism
In transcription, RNA polymerase reads a DNA template and produces an RNA transcript. In eukaryotes the primary transcript is processed: a 5' cap and poly-A tail are added and introns are spliced out. In translation, the ribosome reads the mRNA in triplets (codons). The genetic code has 64 codons, of which 61 specify amino acids and 3 are stop signals; AUG is the usual start codon, and the code is degenerate since most amino acids have more than one codon.

Transfer RNAs carry the amino acids. Each aminoacyl-tRNA synthetase attaches its cognate amino acid to the correct tRNA, and this charging step is a main source of fidelity because the ribosome cannot distinguish a correct amino acid from an incorrectly attached one. Translation proceeds in initiation (assembly of the ribosome at the start codon), elongation (a charged tRNA enters the A site, peptidyl transfer occurs, and the ribosome translocates, with tRNAs moving through A, P, and E sites), and termination (release factors recognize a stop codon). The peptidyl transferase activity resides in ribosomal RNA, so the ribosome is at its core a ribozyme. Adding one amino acid costs roughly four high-energy phosphate bonds in total (two for tRNA charging, two GTP in elongation), which makes translation a major part of a growing cell's energy budget. After synthesis, chaperones assist folding, and post-translational modifications such as phosphorylation, glycosylation, and proteolytic cleavage regulate activity and localization. Regulation operates at every step: transcription factors, microRNAs, and translational control.

## Role in Active Inference
Protein synthesis is not a topic of the free-energy principle proper, but two connections are worth noting. First, it is the machinery by which a cell updates its own structure in response to signals, so the gene regulatory networks that Active Inference treatments of cells discuss ultimately act through it. Second, it is often used as an example of a process with a thermodynamic cost for accuracy: proofreading mechanisms such as kinetic proofreading spend extra energy to lower the error rate below what binding energies alone permit. Describing the ribosome as decoding is a metaphor, and no standard Active Inference model of translation is used in this repository.

## Related concepts
- [[knowledge_base/biology/molecular_biology]] — the broader framework of DNA, RNA, and protein
- [[knowledge_base/biology/gene_regulatory_networks]] — control of which proteins are made and when
- [[knowledge_base/biology/biochemistry]] — enzymatic chemistry of amino acid activation and peptide bond formation
- [[knowledge_base/biology/metabolic_networks]] — energy supply and amino acid precursors
- [[knowledge_base/biology/synthetic_biology]] — engineering expression systems
- [[knowledge_base/biology/cell_biology]] — compartments and organelles where synthesis occurs
