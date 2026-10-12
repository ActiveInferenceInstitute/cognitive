---
title: Topological Data Analysis
description: Methods that use algebraic topology, chiefly persistent homology of simplicial complexes built from point clouds, to summarise the shape of data in a way that is stable under noise.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - topology
  - persistent_homology
  - data_analysis
semantic_relations:
  - type: relates
    links:
      - "[[topology]]"
      - "[[topological_spaces]]"
      - "[[homological_algebra]]"
      - "[[dimension_reduction]]"
      - "[[network_analysis]]"
wikidata: Q4460773
---

# Topological Data Analysis

## Overview
Topological data analysis (TDA) extracts features of the shape of a dataset, such as connected components, loops and voids, that do not depend on the particular coordinates or on small deformations. Its central tool is persistent homology, which records how such features appear and disappear as a scale parameter is varied.

## Construction
A finite point cloud $X$ in a metric space has no interesting topology as a set, so it is thickened to a family of simplicial complexes.
1. **Complex.** The Vietoris-Rips complex $\mathrm{VR}_\varepsilon(X)$ contains a simplex for every subset of points with pairwise distances at most $\varepsilon$. The Čech complex $\mathrm{Cech}_\varepsilon$ is the nerve of the balls of radius $\varepsilon$ around the points, and the two are interleaved: $\mathrm{Cech}_{\varepsilon}\subseteq\mathrm{VR}_{2\varepsilon}\subseteq\mathrm{Cech}_{2\varepsilon}$ in Euclidean space.
2. **Filtration.** Increasing $\varepsilon$ gives nested complexes $K_{\varepsilon_1}\subseteq K_{\varepsilon_2}\subseteq\cdots$.
3. **Homology.** For each dimension $k$, the homology groups $H_k(K_\varepsilon)$ over a field have ranks given by the Betti numbers $\beta_k$: $\beta_0$ counts components, $\beta_1$ independent loops, and $\beta_2$ enclosed voids. The inclusions induce maps between them.
4. **Persistence.** Each class is born at some $\varepsilon_b$ and dies at $\varepsilon_d$. The pairs $(\varepsilon_b,\varepsilon_d)$ are drawn as a barcode or persistence diagram, and long bars are interpreted as robust features while short ones are treated as noise.

The stability theorem of Cohen-Steiner, Edelsbrunner and Harer states that the bottleneck distance between the persistence diagrams of two tame functions is bounded by the supremum norm of their difference, which justifies the noise interpretation.

## Applications and limits
TDA has been applied to neural correlation structure, protein shape, materials and the geometry of learned representations. Computation scales poorly with the number of points and the dimension, and persistence summaries are not complete invariants. Interpreting a feature requires care, because a persistent loop is a statement about the sampled space and not necessarily about a mechanism.

## Relation to Active Inference
The link is that of an analysis tool. Topological summaries can describe the geometry of neural state spaces or of an agent's belief trajectories, and the persistence of cycles has been used to characterise population activity. There is no derivation from the free energy principle. Its relevance to [[hyperspace_ontology]] is as a way to describe the shape of high-dimensional conceptual spaces.

## Related concepts
- [[topology]] — the field from which the invariants come
- [[topological_spaces]] — the underlying definitions of continuity and connectedness
- [[homological_algebra]] — the algebra of chain complexes behind homology
- [[dimension_reduction]] — a complementary way to study high-dimensional data
- [[network_analysis]] — graph data also admit clique-complex persistence

## References
- Edelsbrunner, H., Letscher, D., & Zomorodian, A. (2002). Topological persistence and simplification. Discrete & Computational Geometry, 28, 511-533.
- Carlsson, G. (2009). Topology and data. Bulletin of the American Mathematical Society, 46, 255-308.
- Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). Stability of persistence diagrams. Discrete & Computational Geometry, 37, 103-120.
- Edelsbrunner, H., & Harer, J. (2010). Computational Topology: An Introduction. American Mathematical Society.
