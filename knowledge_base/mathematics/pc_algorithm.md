---
title: PC Algorithm
description: The Peter-Clark constraint-based algorithm that recovers a Markov equivalence class of causal graphs from conditional independence tests, with its assumptions, steps and complexity.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - causal_discovery
  - bayesian_networks
  - structure_learning
semantic_relations:
  - type: relates
    links:
      - "[[bayesian_networks]]"
      - "[[conditional_independence_tests]]"
      - "[[d_separation]]"
      - "[[directed_acyclic_graphs]]"
      - "[[causal_inference]]"
      - "[[faithfulness]]"
---

# PC Algorithm

## Overview
The PC algorithm, named for its authors Peter Spirtes and Clark Glymour (1991), learns the structure of a directed acyclic graph from observational data using only conditional independence tests. It does not return a single graph. It returns a representative of the Markov equivalence class, the set of graphs that imply the same conditional independences, drawn as a partially directed graph.

## Procedure
The algorithm assumes the causal Markov condition, faithfulness (every independence in the distribution is implied by $d$-separation in the true graph), causal sufficiency (no unmeasured common causes) and a correct independence test. It proceeds in three stages.

1. **Skeleton.** Start with the complete undirected graph on the variables. For increasing conditioning-set size $k=0,1,2,\dots$, remove the edge between $X$ and $Y$ if some set $S$ of $k$ neighbours of $X$ (or of $Y$) satisfies $X\perp\!\!\!\perp Y\mid S$, and record $S$ as the separating set of the pair.
2. **V-structures.** For each unshielded triple $X - Z - Y$ in which $X$ and $Y$ are not adjacent, orient $X\to Z\leftarrow Y$ if $Z$ is not in the separating set of $X$ and $Y$.
3. **Propagation.** Apply orientation rules (Meek's rules) repeatedly to direct further edges whenever the opposite orientation would create a new v-structure or a directed cycle.

Edges left undirected reflect real ambiguity: for instance $X\to Y\to Z$, $X\leftarrow Y\to Z$ and $X\leftarrow Y\leftarrow Z$ imply the same independences and cannot be told apart from observational data alone. The output is therefore a completed partially directed acyclic graph.

In the worst case the number of independence tests is exponential in the number of variables, but for sparse graphs with bounded maximum degree it is polynomial. Kalisch and Bühlmann (2007) proved consistency of a version of the algorithm for sparse high-dimensional Gaussian data. Errors in early tests propagate to the orientation stage, and results can depend on variable order. Extensions such as FCI handle latent confounders.

## Relation to Active Inference
Learning model structure is a topic in Active Inference, where it is usually done by Bayesian model comparison and reduction, scoring alternative structures by their evidence. The PC algorithm is a constraint-based alternative that needs no priors over graphs. The connection is conceptual, since both seek generative structure from data; constraint-based discovery is not part of the standard formulation.

## Related concepts
- [[bayesian_networks]] — the graphical models whose structure is recovered
- [[conditional_independence_tests]] — the statistical tests the algorithm calls
- [[d_separation]] — the graphical criterion linking graphs and independences
- [[faithfulness]] — the assumption making independences identify structure
- [[causal_inference]] — the goal of causal discovery
- [[intervention_calculus]] — what the learned graph is used for

## References
- Spirtes, P., & Glymour, C. (1991). An algorithm for fast recovery of sparse causal graphs. Social Science Computer Review, 9, 62-72.
- Spirtes, P., Glymour, C., & Scheines, R. (2000). Causation, Prediction, and Search (2nd ed.). MIT Press.
- Kalisch, M., & Bühlmann, P. (2007). Estimating high-dimensional directed acyclic graphs with the PC-algorithm. Journal of Machine Learning Research, 8, 613-636.
