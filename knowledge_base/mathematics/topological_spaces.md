---
title: Topological Spaces
description: Sets equipped with a topology of open sets that formalises continuity, convergence and nearness, with standard properties, examples and the category Top of spaces and continuous maps.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - topology
  - continuity
  - category_theory
semantic_relations:
  - type: relates
    links:
      - "[[topology]]"
      - "[[algebraic_topology]]"
      - "[[metric_geometry]]"
      - "[[normed_spaces]]"
      - "[[category_theory]]"
---

# Topological Spaces

## Overview
A topological space is a set $X$ together with a collection $\tau$ of subsets, called the open sets, satisfying three conditions: $\emptyset$ and $X$ belong to $\tau$; any union of members of $\tau$ is in $\tau$; and any finite intersection of members of $\tau$ is in $\tau$. The collection $\tau$ is a topology on $X$. The structure is the minimal one needed to define continuity, limits and connectedness without reference to distances.

## Basic notions
- **Continuity.** A function $f:X\to Y$ between topological spaces is continuous if the preimage of every open set is open. A homeomorphism is a continuous bijection with continuous inverse, and homeomorphic spaces are regarded as topologically the same.
- **Compactness.** A space is compact if every open cover has a finite subcover. A continuous image of a compact space is compact, and a continuous real-valued function on a compact space attains its maximum.
- **Connectedness.** A space is connected if it is not the union of two disjoint nonempty open sets.
- **Separation.** In a Hausdorff space, any two distinct points have disjoint open neighbourhoods, which guarantees unique limits of convergent sequences.

## Examples
- **Metric spaces.** Open balls generate the metric topology. Normed spaces, and therefore Euclidean space, are special cases ([[normed_spaces]]).
- **Not all topologies come from metrics.** The Zariski topology of algebraic geometry is an example of a topology that is not Hausdorff, and so not metrisable ([[algebraic_geometry]]).

The spaces and continuous maps form a category, usually written $\mathbf{Top}$ ([[category_theory]]). The properties that survive homeomorphism are studied in [[topology]], and invariants such as homotopy and homology groups in [[algebraic_topology]] (Munkres, 2000).

## Relation to Active Inference
Topological spaces supply the general setting for state spaces and for notions such as the convergence of beliefs and continuity of generative maps. Topological and categorical formulations of the Markov blanket and the free energy principle use them as background structure, as do treatments of the space of probability measures under weak convergence ([[weak_convergence]]). Most Active Inference work uses finite or Euclidean spaces, where the general definition is rarely invoked, so this is a foundational link rather than a result specific to the theory.

## Related concepts
- [[topology]] — the study of properties preserved by continuous deformation
- [[algebraic_topology]] — invariants of spaces
- [[metric_geometry]] — spaces with a distance
- [[normed_spaces]] — vector spaces with a norm-induced topology
- [[category_theory]] — the framework of which Top is an example
- [[weak_convergence]] — a topology on measures

## References
- Munkres, J. R. (2000). Topology (2nd ed.). Prentice Hall.
- Willard, S. (1970). General Topology. Addison-Wesley.
