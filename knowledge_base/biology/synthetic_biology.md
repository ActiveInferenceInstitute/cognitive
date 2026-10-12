---
title: Synthetic Biology
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - synthetic_biology
  - genetic_circuits
  - bioengineering
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/biology/gene_regulatory_networks]]"
      - "[[knowledge_base/biology/molecular_biology]]"
      - "[[knowledge_base/biology/systems_biology]]"
wikidata: Q862838
---

# Synthetic Biology

## Overview
Synthetic biology applies engineering principles to biology: designing and building new biological parts, devices, and systems, or re-designing existing ones, with the aim of predictable function. It differs from classical genetic engineering by its emphasis on standardization, modularity, and quantitative design-build-test cycles.

## Mechanism
Two landmark 2000 papers established the field's circuit-level approach. Gardner, Cantor and Collins built a genetic toggle switch from two mutually repressing genes in Escherichia coli, giving a bistable memory element. Elowitz and Leibler built the repressilator, a ring of three repressors producing sustained oscillations in single cells. Later work added logic gates, sensors, and feedback controllers, assembled from parts such as promoters, ribosome binding sites, and coding sequences. The BioBrick standard and the iGEM competition promoted part reuse, and DNA assembly methods (Gibson assembly, Golden Gate) made constructing larger systems routine. Genome editing with CRISPR-Cas systems provides targeted modification, and DNA synthesis cost reductions have made whole-gene design practical.

Top-down efforts reduce an existing genome: the JCVI-syn3.0 minimal cell, reported by Hutchison and colleagues in 2016, has a 473-gene genome, and a notable fraction of those genes had unknown function. Bottom-up efforts assemble cell-like systems from molecular components, including cell-free expression systems and liposome-encapsulated reactions. Persistent difficulties are context dependence (a part behaves differently in different hosts or genetic backgrounds), metabolic burden on the host, noise in gene expression, and evolutionary instability, since engineered circuits that cost fitness are lost under selection.

## Role in Active Inference
The connection is conceptual rather than methodological. Gene regulatory circuits process signals and maintain states, and some authors describe cells as performing inference about their environment; synthetic circuits make such claims testable by building minimal systems with known wiring. Designing a circuit to track an environmental variable is a controllable instance of a system that estimates and acts. Beyond this analogy, Active Inference is not a standard tool in synthetic biology design, and this repository does not implement circuit models.

## Related concepts
- [[knowledge_base/biology/gene_regulatory_networks]] — the natural circuits that synthetic designs imitate
- [[knowledge_base/biology/molecular_biology]] — the underlying DNA, RNA, and protein machinery
- [[knowledge_base/biology/protein_synthesis]] — expression machinery that circuits depend on
- [[knowledge_base/biology/autopoiesis]] — self-producing organization, relevant to minimal and artificial cells
- [[knowledge_base/biology/cellular_intelligence]] — information processing in single cells
- [[knowledge_base/systems/artificial_life]] — the computational and wet-lab study of life-like systems

## References
- Gardner, T. S., Cantor, C. R., and Collins, J. J. (2000). Construction of a genetic toggle switch in Escherichia coli. Nature.
- Elowitz, M. B., and Leibler, S. (2000). A synthetic oscillatory network of transcriptional regulators. Nature.
- Hutchison, C. A., et al. (2016). Design and synthesis of a minimal bacterial genome. Science.
