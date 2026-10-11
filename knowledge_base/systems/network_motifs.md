---
title: Network Motifs
description: Small recurring patterns of interconnection that occur in real networks more often than in randomised ones, such as the feed-forward loop in gene regulation, and their functional interpretation.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - network_motifs
  - systems_biology
  - graph_theory
semantic_relations:
  - type: relates
    links:
      - "[[gene_regulatory_networks]]"
      - "[[transcription_factors]]"
      - "[[network_theory]]"
      - "[[systems_biology]]"
      - "[[feedback_loops]]"
---

# Network Motifs

## Overview
Network motifs are small subgraphs, typically of three or four nodes, that appear in a real network significantly more often than in randomised networks with the same number of nodes and the same degree of each node. The idea was introduced by Milo and colleagues in 2002, who found shared motifs in networks as different as gene regulation, food webs and electronic circuits. The working hypothesis is that over-represented patterns are building blocks that perform specific information-processing functions and were retained by selection.

## Detection
A motif is identified by counting occurrences of each possible subgraph pattern in the observed network, generating an ensemble of randomised networks that preserve local properties, and comparing counts. A pattern whose count is high relative to the ensemble, usually judged by a $Z$-score or $p$-value, is called a motif. The result depends on the choice of null model, so over-representation alone does not establish function.

## Examples in gene regulation
In transcriptional networks, where nodes are genes and edges are regulation by [[transcription_factors]], common motifs include:
- **Negative autoregulation**, in which a factor represses its own production. It speeds up the response time and reduces cell-to-cell variability in expression.
- **Feed-forward loop**, in which factor X regulates Y, and both X and Y regulate Z. In the coherent type-1 form with AND logic, Z is activated only when X has been on for long enough, so the circuit filters out brief input pulses.
- **Single-input module**, in which one regulator controls a group of genes, allowing coordinated timing.

Such functions have been analysed with dynamic models and tested by experiments on bacterial gene circuits.

## Relation to Active Inference
The link is an analogy at the level of structure rather than a derivation. Active Inference models can be written as graphs, with message passing between hidden states and observations, and motif-like analysis can in principle be applied to the connectivity of such models or of cortical circuits. Canonical microcircuit proposals for predictive coding assign particular connection patterns to prediction and error units. Whether these recurring patterns are motifs in the statistical sense above has not been established.

## Related concepts
- [[gene_regulatory_networks]] — where motifs were analysed most thoroughly
- [[transcription_factors]] — the regulators that form motif edges
- [[network_theory]] — graph-theoretic background
- [[feedback_loops]] — autoregulation and other loop motifs
- [[metabolic_networks]] — another class of networks analysed for motifs
- [[systems_biology]] — the field that uses motifs as design principles

## References
- Milo, R., Shen-Orr, S., Itzkovitz, S., Kashtan, N., Chklovskii, D., & Alon, U. (2002). Network motifs: simple building blocks of complex networks. Science, 298, 824-827.
- Alon, U. (2007). Network motifs: theory and experimental approaches. Nature Reviews Genetics, 8, 450-461.
- Alon, U. (2006). An Introduction to Systems Biology: Design Principles of Biological Circuits. Chapman and Hall/CRC.
