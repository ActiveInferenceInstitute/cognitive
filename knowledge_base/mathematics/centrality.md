---
title: Centrality
description: Quantitative measures of node importance in a network, including degree, closeness, betweenness, eigenvector and PageRank centrality, their definitions, limitations, and use in network neuroscience.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - graph_theory
  - network_analysis
  - complex_systems
semantic_relations:
  - type: relates
    links:
      - "[[graph_theory]]"
      - "[[network_analysis]]"
      - "[[network_neuroscience]]"
      - "[[scale_free_networks]]"
      - "[[spectral_theory]]"
wikidata: Q5062194
---

# Centrality

## Overview
Centrality refers to a family of measures that assign each node of a graph a number indicating its structural importance. No single measure is correct for all purposes, since "importance" depends on what flows through the network and on how it flows. Let $G = (V, E)$ be a graph with $n$ nodes and adjacency matrix $A$, where $A_{ij} = 1$ if there is an edge from $i$ to $j$.

## Common measures
- **Degree centrality.** $c_D(v) = \deg(v) = \sum_j A_{vj}$, often divided by $n-1$ to lie in $[0,1]$. It counts direct connections.
- **Closeness centrality.** $c_C(v) = (n-1) / \sum_{u \neq v} d(v, u)$, where $d$ is the shortest-path distance. It is high for nodes that are near all others.
- **Betweenness centrality.** $c_B(v) = \sum_{s \neq v \neq t} \sigma_{st}(v) / \sigma_{st}$, where $\sigma_{st}$ is the number of shortest paths from $s$ to $t$ and $\sigma_{st}(v)$ the number of those passing through $v$. It measures how often a node lies on routes between others (Freeman, 1978).
- **Eigenvector centrality.** $x_v = \lambda^{-1}\sum_u A_{vu} x_u$, that is $Ax = \lambda x$, with $x$ taken as the leading eigenvector (for a directed graph this convention scores out-links; the in-link version uses $A^\top$). A node scores highly if it is connected to other high-scoring nodes (Bonacich, 1972). For a connected undirected graph the Perron-Frobenius theorem guarantees a nonnegative leading eigenvector.
- **PageRank.** A variant for directed graphs in which a random walker follows a link with probability $d$ and jumps to a random node with probability $1-d$, giving $p = \frac{1-d}{n}\mathbf{1} + d\,M p$, where $M$ is the column-normalised link matrix.

## Limitations
The measures can disagree, so the ranking of nodes depends on the choice. Betweenness and closeness assume that flow follows shortest paths, which is not true of diffusion or random walks, and many are sensitive to missing or noisy edges. In networks with heavy-tailed degree distributions (see [[scale_free_networks]]), a few hubs can dominate the rankings (Newman, 2018).

## Relation to Active Inference
Centrality is a descriptive tool and has no special status in Active Inference. It is applied in network neuroscience to identify hub regions in structural and functional connectomes, where nodes of high degree or betweenness are interpreted as integrating information (see [[network_neuroscience]]). Message passing schemes for inference on graphical models run on graphs whose structure could be analysed in the same way, but this is a use of the tool, not a statement of the theory.

## Related concepts
- [[graph_theory]] — the mathematical setting
- [[network_analysis]] — the applied practice that uses centrality
- [[network_neuroscience]] — an application to brain connectivity
- [[scale_free_networks]] — networks where hubs dominate
- [[spectral_theory]] — the basis of eigenvector-based measures
- [[small_world]] — a topology in which shortcuts raise centrality of some nodes

## References
- Freeman, L. C. (1978). Centrality in social networks: conceptual clarification. Social Networks, 1, 215-239.
- Bonacich, P. (1972). Factoring and weighting approaches to status scores and clique identification. Journal of Mathematical Sociology, 2, 113-120.
- Newman, M. E. J. (2018). Networks (2nd ed.). Oxford University Press.
