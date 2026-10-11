---
title: Chern Classes
description: Topological invariants of complex vector bundles, defined by axioms or by curvature forms, with Chern numbers as integers, and an honest assessment of their remote relation to Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - topology
  - differential_geometry
  - characteristic_classes
semantic_relations:
  - type: relates
    links:
      - "[[differential_geometry]]"
      - "[[algebraic_topology]]"
      - "[[topology]]"
      - "[[information_geometry]]"
      - "[[free_energy_principle]]"
---

# Chern Classes

## Overview
Chern classes are characteristic classes of complex vector bundles. To a complex vector bundle $E\to M$ over a manifold they assign cohomology classes $c_k(E)\in H^{2k}(M;\mathbb{Z})$ that measure how far the bundle is from being a trivial product. They are invariant under isomorphism of bundles, so they give a way to tell bundles apart by topological data alone.

## Definition
The total Chern class is $c(E)=1+c_1(E)+c_2(E)+\dots$, and it is determined by a few axioms.
- **Naturality.** For a map $f$, $c(f^*E)=f^*c(E)$.
- **Whitney sum formula.** $c(E\oplus F)=c(E)\,c(F)$.
- **Normalisation.** $c_1$ of the tautological line bundle over $\mathbb{CP}^1$ generates $H^2(\mathbb{CP}^1;\mathbb{Z})$, up to a sign convention.
- **Vanishing.** $c_k(E)=0$ for $k$ larger than the rank of $E$.

In Chern-Weil theory the classes are computed from a connection with curvature two-form $F$: $c(E)$ is represented by the de Rham class of $\det\!\big(I+\tfrac{i}{2\pi}F\big)$. Since this class does not depend on the chosen connection, local geometric data yields a global topological invariant. For a complex line bundle $L$, $c_1(L)$ classifies the bundle, and its integral over a closed oriented surface is an integer, the Chern number or degree.

## Uses
Chern classes appear in algebraic geometry, in the classification of vector bundles, and in index theorems. In physics, integer Chern numbers of Berry curvature account for the quantisation of Hall conductance in the integer quantum Hall effect and classify topological band structures.

## Relation to Active Inference
No established derivation or active research programme connects Chern classes to Active Inference. The loose connection is that the [[free_energy_principle]] literature, and [[information_geometry]], use differential-geometric objects such as connections and curvature on statistical manifolds. That use does not currently involve complex vector bundles or their characteristic classes, so any role for them would be speculative.

## Related concepts
- [[differential_geometry]] — connections and curvature, from which Chern classes are computed
- [[algebraic_topology]] — the cohomology groups in which the classes live
- [[topology]] — global invariants that survive continuous deformation
- [[information_geometry]] — geometric structure on spaces of probability distributions
- [[free_energy_principle]] — the framework in which geometric language is sometimes invoked

## References
- Milnor, J. W., & Stasheff, J. D. (1974). Characteristic Classes. Princeton University Press.
- Nakahara, M. (2003). Geometry, Topology and Physics (2nd ed.). Institute of Physics Publishing.
- Bott, R., & Tu, L. W. (1982). Differential Forms in Algebraic Topology. Springer.
