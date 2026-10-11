---
title: Small World
description: Network architecture combining high local clustering with short average path length between nodes, its Watts-Strogatz model, and its role in brain connectivity and Active Inference models.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - networks
  - small_world
  - connectivity
semantic_relations:
  - type: relates
    links:
      - "[[network_theory]]"
      - "[[scale_free_networks]]"
      - "[[network_neuroscience]]"
      - "[[graph_theory]]"
      - "[[network_models]]"
---

# Small World

## Overview
A small-world network is one in which most nodes are not neighbours of one another, yet any node can be reached from any other through a small number of steps. Formally it combines a high clustering coefficient $C$, the probability that two neighbours of a node are themselves linked, like that of a regular lattice, with an average shortest path length $L$ that grows only logarithmically with the number of nodes $N$, like that of a random graph. The phrase is borrowed from the observation that acquaintance chains between people are surprisingly short.

## The Watts-Strogatz model
Watts and Strogatz started from a ring lattice in which each node links to its $k$ nearest neighbours, then rewired each link at random with probability $p$. At $p=0$ the graph is highly clustered with long paths, and at $p=1$ it is random with short paths and low clustering. For intermediate $p$ a small fraction of long-range shortcuts sharply reduces $L$ while $C$ stays high. This showed that very little randomness is needed to produce short paths. A network is commonly called small-world when $L\approx L_{\mathrm{rand}}$ and $C\gg C_{\mathrm{rand}}$, compared with a random graph of equal size and density.

## Occurrence and functional meaning
Small-world properties have been reported in neural wiring of the nematode worm, in structural and functional connectivity of the mammalian brain, in power grids, and in collaboration and social networks. In brain networks the architecture is interpreted as balancing segregated local processing, supported by clustering, against integrated global communication, supported by short paths, at a modest wiring cost. These findings depend on the thresholding and parcellation used to build the network, so quantitative comparisons need care.

## Relation to Active Inference
The link is an analogy supported by shared structure, not a derivation. Hierarchical generative models require dense local interaction within a level and efficient long-range exchange between levels, which is the kind of organisation small-world graphs provide. Studies of effective connectivity in the brain, including those using dynamic causal modelling, describe network topology that is consistent with this. Active Inference does not itself predict small-world structure.

## Related concepts
- [[network_theory]] — the general framework in which clustering and path length are defined
- [[scale_free_networks]] — heavy-tailed degrees, often discussed alongside small worlds
- [[network_neuroscience]] — applies small-world analysis to brain networks
- [[graph_theory]] — formal definitions of paths, clusters and degree
- [[network_models]] — rewiring and growth models that generate such graphs
- [[connectivity]] — structural and functional links between brain regions

## References
- Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. Nature, 393, 440-442.
- Bassett, D. S., & Bullmore, E. (2006). Small-world brain networks. The Neuroscientist, 12, 512-523.
