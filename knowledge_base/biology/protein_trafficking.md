---
title: Protein Trafficking
description: How newly made proteins are sorted and delivered to organelles, membranes or the cell exterior, including signal sequences, vesicle transport, and the relevance to synaptic receptor placement.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - cell_biology
  - membrane_transport
  - protein_sorting
semantic_relations:
  - type: relates
    links:
      - "[[vesicle_traffic]]"
      - "[[protein_synthesis]]"
      - "[[cell_biology]]"
      - "[[neural_plasticity]]"
      - "[[receptor_phosphorylation]]"
---

# Protein Trafficking

## Overview
Protein trafficking, or protein sorting, is the set of processes that direct newly synthesised proteins to the compartment where they function: the nucleus, mitochondria, endoplasmic reticulum, Golgi apparatus, lysosomes, plasma membrane or extracellular space. Cells need these processes because most proteins are made on ribosomes in the cytosol, whereas their working sites are often separated from it by membranes.

## Sorting signals and routes
Destination is generally encoded in the protein itself, as short sequence motifs or as modifications. An N-terminal signal sequence directs proteins to the endoplasmic reticulum, a nuclear localisation signal allows import through nuclear pores, and other sequences target mitochondria or peroxisomes (Alberts et al., 2015). Three broad mechanisms are recognised: gated transport through pores, translocation across a membrane by protein machinery, and vesicular transport, in which cargo-laden membrane carriers bud from one compartment and fuse with the next. The secretory pathway runs from endoplasmic reticulum through Golgi to the surface, with quality control that retains misfolded proteins, and a retrieval route returns resident proteins. Endocytic routes bring surface proteins in for recycling or degradation. The vesicle step is treated in [[vesicle_traffic]].

## Role in neurons
Neurons are large, polarised cells, so trafficking has special weight. Synaptic proteins are made in the cell body or locally and must be transported along axons and dendrites by motor proteins on microtubules. The number of receptors at a synapse depends on exocytosis, endocytosis and lateral movement, and changes in this balance are central to models of synaptic strengthening and weakening (see [[neural_plasticity]] and [[synaptic_plasticity]]). Phosphorylation of receptors and associated proteins can alter their sorting (see [[receptor_phosphorylation]]).

## Relation to Active Inference
The link is indirect. Trafficking is a cellular mechanism by which synaptic weights, which implementations of learning in neural models treat as parameters, are physically adjusted. Active Inference itself does not specify this mechanism, and mapping precision or connection strength onto receptor numbers is an interpretation rather than a derivation.

## Related concepts
- [[vesicle_traffic]] — the carrier-based transport step of sorting
- [[protein_synthesis]] — the upstream process producing the cargo
- [[protein_folding]] — folding state governs quality control in the secretory pathway
- [[synaptic_plasticity]] — receptor trafficking underlies changes in synaptic strength
- [[neural_plasticity]] — the wider framework in which synaptic changes are studied

## References
- Alberts, B., et al. (2015). Molecular Biology of the Cell (6th ed.). Garland Science.
- Rothman, J. E. (1994). Mechanisms of intracellular protein transport. Nature, 372, 55-63.
- Palade, G. (1975). Intracellular aspects of the process of protein synthesis. Science, 189, 347-358.
