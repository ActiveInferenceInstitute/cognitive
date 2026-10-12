---
title: Sheaf Theory
description: "Presheaves and sheaves as tools for passing from local data to consistent global data, with cellular sheaves on graphs, cohomology, and speculative links to distributed inference."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - topology
  - category_theory
  - local_to_global
semantic_relations:
  - type: relates
    links:
      - "[[topology]]"
      - "[[homological_algebra]]"
      - "[[category_theory]]"
      - "[[factor_graphs]]"
wikidata: Q77827144
---

# Sheaf Theory

## Overview
A sheaf organises local data attached to the open sets of a space so that compatible local pieces assemble uniquely into global data. It was developed (Leray, Cartan, Grothendieck) for algebraic topology and geometry, and recently adopted for data fusion, networks and contextuality.

## Definition
**Presheaf.** A contravariant functor $\mathcal F$ from the poset of open sets of $X$ to sets (or groups, vector spaces): to each open $U$ it assigns $\mathcal F(U)$, and to each inclusion $V\subseteq U$ a restriction map $\mathcal F(U)\to\mathcal F(V)$, compatible with composition.

**Sheaf.** A presheaf satisfying two axioms for every open cover $\{U_i\}$ of $U$: (locality) sections that agree on all $U_i$ are equal; (gluing) sections $s_i\in\mathcal F(U_i)$ agreeing on overlaps $U_i\cap U_j$ come from a unique section on $U$. Continuous real functions form a sheaf; bounded functions do not (boundedness is not local).

**Stalks and cohomology.** The stalk $\mathcal F_x$ is the colimit of $\mathcal F(U)$ over neighbourhoods of $x$. Sheaf cohomology $H^k(X;\mathcal F)$ measures obstructions to extending local solutions globally, using the derived-functor machinery of [[homological_algebra]]. $H^0$ is the space of global sections.

**Cellular sheaves.** On a graph, a cellular sheaf assigns vector spaces to vertices and edges with linear restriction maps; global sections are the consistent assignments, and the sheaf Laplacian (Hansen and Ghrist, 2019) measures inconsistency.

## Role in Active Inference
The connection is suggestive and has not been established as a derivation. Inference on a [[factor_graphs|factor graph]] has the sheaf-like shape of local beliefs that must agree on shared variables, and consistency of marginals is a gluing condition. Abramsky and Brandenburger (2011) used sheaves to formalise contextuality, a setting in which local probability assignments cannot be glued, which is relevant to arguments about when a joint generative model exists. Category-theoretic treatments of compositional Active Inference use related functorial machinery. Standard implementations (belief propagation, variational message passing) do not need sheaf language, and claims that nested Markov blankets form a sheaf should be read as proposals.

## Related concepts
- [[topology]] — the spaces whose open sets index a sheaf
- [[category_theory]] — functors, limits and colimits underlying the definition
- [[belief_propagation]] — local message exchange seeking global consistency
- [[probabilistic_graphical_models]] — graph-structured local models with global consistency

## References
- Abramsky, S. and Brandenburger, A. (2011). The sheaf-theoretic structure of non-locality and contextuality. New Journal of Physics.
- Hansen, J. and Ghrist, R. (2019). Toward a spectral theory of cellular sheaves. Journal of Applied and Computational Topology.
