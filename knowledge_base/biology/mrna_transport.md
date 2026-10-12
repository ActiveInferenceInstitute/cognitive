---
title: mRNA Transport
description: The targeting of messenger RNA molecules to specific cellular locations, including neuronal dendrites and synapses, enabling local protein synthesis that supports synaptic plasticity and memory.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - molecular_biology
  - neuroscience
  - plasticity
semantic_relations:
  - type: relates
    links:
      - "[[protein_synthesis]]"
      - "[[protein_trafficking]]"
      - "[[gene_expression]]"
      - "[[neural_plasticity]]"
      - "[[synaptic_tagging]]"
      - "[[late_ltp]]"
---

# mRNA Transport

## Overview
After transcription in the nucleus, many messenger RNAs (mRNAs) are exported to the cytoplasm and translated near their site of origin. Some are instead transported to particular locations, where they are held in a repressed state until a signal triggers translation. Localising mRNAs allows a cell to produce a protein where it is needed, and to respond to local signals more quickly than a protein shipped from the cell body could. Martin and Ephrussi (2009) review the process across cell types, including the classic examples of localised mRNAs in the Drosophila oocyte.

## Mechanism
Transport depends on sequence or structural elements in the mRNA, often in the 3' untranslated region, sometimes called zipcodes, which are recognised by RNA-binding proteins. The mRNA and its binding proteins assemble into ribonucleoprotein particles, often called RNA granules, that are carried along microtubules by the motor proteins kinesin and dynein and along actin filaments by myosins. During transit, translation is generally repressed. At the destination, synaptic activity can release the repression and allow local translation.

## In neurons
Neurons are extended cells with dendrites and axons that can be far from the nucleus. Ribosomes were observed under dendritic spines by Steward and Levy (1982), which suggested that proteins could be made locally, and many mRNAs have since been found in dendrites and axons (Holt and Schuman, 2013). Examples reported to be targeted to dendrites include the mRNAs for CaMKIIα and Arc. Local translation is thought to contribute to the protein-synthesis-dependent phase of long-term potentiation and to synaptic tagging, in which a synapse marked by recent activity captures newly made proteins (see [[late_ltp]] and [[synaptic_tagging]]). Bramham and Wells (2007) review the transport and translation of dendritic mRNAs. Defects in RNA transport and local translation are implicated in some neurodevelopmental and neurodegenerative disorders.

## Relation to Active Inference
There is no direct link to Active Inference. mRNA transport belongs to the molecular substrate of learning, and may be relevant when abstract rules for parameter updating are compared with how synapse-specific changes are actually implemented. Any such mapping is speculative at present.

## Related concepts
- [[protein_synthesis]] — the translation that transported mRNAs undergo
- [[protein_trafficking]] — the parallel transport of proteins within cells
- [[late_ltp]] — the protein-synthesis-dependent phase of potentiation
- [[synaptic_tagging]] — a proposal for how synapses capture new proteins
- [[dendritic_computation]] — the integrative properties of dendrites that local synthesis may adjust

## References
- Steward, O., & Levy, W. B. (1982). Preferential localization of polyribosomes under the base of dendritic spines in granule cells of the dentate gyrus. Journal of Neuroscience, 2, 284-291.
- Martin, K. C., & Ephrussi, A. (2009). mRNA localization: gene expression in the spatial dimension. Cell, 136, 719-730.
- Bramham, C. R., & Wells, D. G. (2007). Dendritic mRNA: transport, translation and function. Nature Reviews Neuroscience, 8, 776-789.
- Holt, C. E., & Schuman, E. M. (2013). The central dogma decentralized: new perspectives on RNA function and local translation in neurons. Neuron, 80, 648-657.
