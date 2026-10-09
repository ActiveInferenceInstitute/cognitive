---
title: Network Models
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - network_models
  - graph_theory
  - small_world
  - scale_free
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/systems/network_theory]]"
      - "[[knowledge_base/mathematics/network_science]]"
      - "[[knowledge_base/mathematics/graph_theory]]"
      - "[[knowledge_base/biology/ecological_networks]]"
---

# Network Models

## Overview
Network models are generative or statistical descriptions of how graphs with nodes and links are formed and how they behave. They are used to capture the structure of systems such as food webs, gene regulatory circuits, neural connectomes, and social interaction patterns, and to test which formation rules reproduce observed structure.

## Mechanism
Classical random graphs (Erdos and Renyi 1959) connect each pair of nodes independently with probability p; their degree distribution is binomial (approximately Poisson), clustering is low, and a giant connected component appears when the mean degree exceeds 1. Real networks usually differ. The Watts-Strogatz model (1998) starts from a ring lattice and rewires a small fraction of links, producing high clustering together with short average path length, the small-world property. The Barabasi-Albert model (1999) grows a network by preferential attachment, where new nodes link to existing nodes in proportion to their degree, yielding a heavy-tailed degree distribution with hubs; whether many real networks are strictly scale-free is disputed. Other models include the configuration model (fixed degree sequence), the stochastic block model (group-dependent link probabilities, used to infer community structure), exponential random graph models, and multilayer and temporal networks.

Common descriptors are degree distribution, clustering coefficient, path length, centrality, modularity, and nestedness (used for plant-pollinator and host-parasite webs). Dynamics on networks include epidemic spreading (SIR), opinion dynamics, synchronization, and cascading failure; structure shapes outcomes, as when hubs lower epidemic thresholds and increase vulnerability to targeted attack. Interaction networks within ant colonies, measured by tracking individuals, show that contact structure is heterogeneous and tied to task, which affects both information flow and disease spread.

## Role in Active Inference
Two connections should be kept apart. First, probabilistic graphical models used in Active Inference, such as Bayesian networks and factor graphs, are network models whose edges are conditional dependencies, and inference runs by passing messages along them (see [[knowledge_base/systems/message_passing_networks]]). Second, in the free-energy formulation of a particular physics (Friston 2019), the Markov blanket partitions a system into internal, external, sensory, and active states, which is a statement about the sparse coupling structure of a network. Using empirical network models, such as small-world or scale-free graphs, as priors on the connectivity of agent societies is a modeling choice available in multi-agent work and is not a result of the theory.

## Related concepts
- [[knowledge_base/systems/network_theory]] — general network concepts and measures
- [[knowledge_base/mathematics/network_science]] — mathematical treatment of graphs and dynamics
- [[knowledge_base/biology/ecological_networks]] — food webs and mutualistic networks
- [[knowledge_base/biology/gene_regulatory_networks]] — regulatory graphs inside cells
- [[knowledge_base/free_energy_principle/systems/network_dynamics]] — dynamics on networks under the free-energy principle
- [[knowledge_base/systems/complex_adaptive_systems]] — adaptive agents arranged on networks

## References
- Erdos, P., and Renyi, A. (1959). On random graphs I. Publicationes Mathematicae.
- Watts, D. J., and Strogatz, S. H. (1998). Collective dynamics of small-world networks. Nature.
- Barabasi, A.-L., and Albert, R. (1999). Emergence of scaling in random networks. Science.
- Newman, M. E. J. (2010). Networks: An Introduction.
