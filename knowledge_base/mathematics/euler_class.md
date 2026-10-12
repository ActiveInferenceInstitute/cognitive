---
title: Euler Class
description: A characteristic class of oriented real vector bundles that lives in the cohomology of the base in the degree of the rank, vanishes when the bundle has a nowhere-zero section, and integrates to the Euler characteristic for the tangent bundle.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - characteristic_classes
  - algebraic_topology
  - vector_bundles
semantic_relations:
  - type: relates
    links:
      - "[[chern_classes]]"
      - "[[euler_characteristic]]"
      - "[[fiber_bundles]]"
      - "[[algebraic_topology]]"
      - "[[differential_geometry]]"
---

# Euler Class

## Overview
The Euler class is a characteristic class: a cohomology class attached to an oriented real vector bundle that is natural under pullback and measures a kind of twisting of the bundle. For an oriented real vector bundle $E\to B$ of rank $n$ it is a class $e(E)\in H^n(B;\mathbb{Z})$. Its main role is as the obstruction to finding a nowhere-vanishing section, and, for the tangent bundle of a closed oriented manifold, as the cohomological form of the Euler characteristic. The standard references are Milnor and Stasheff (1974) and Bott and Tu (1982).

## Construction
Let $E$ be oriented of rank $n$. The orientation determines a Thom class $U\in H^n(E,E\setminus 0;\mathbb{Z})$ that restricts to a generator on each fibre. Pulling $U$ back along the zero section $s:B\to E$, and mapping to absolute cohomology, defines $e(E)=s^*U\in H^n(B;\mathbb{Z})$. When $B$ is a closed oriented manifold, $e(E)$ is also the Poincaré dual of the zero set of a generic section.

## Properties
- **Naturality.** $e(f^*E)=f^*e(E)$ for any map $f$.
- **Whitney product formula.** $e(E\oplus F)=e(E)\smile e(F)$.
- **Orientation.** Reversing the orientation of $E$ changes the sign of $e(E)$. Consequently, if $n$ is odd then $2e(E)=0$.
- **Sections.** If $E$ has a nowhere-vanishing section then $e(E)=0$. The converse holds in general only under extra hypotheses, for example when the rank equals the dimension of the base.
- **Relation to Chern classes.** For a complex vector bundle of complex rank $k$, regarded as an oriented real bundle of rank $2k$, the Euler class equals the top Chern class $c_k$ (see [[chern_classes]]).

## Euler characteristic and curvature
For a closed oriented manifold $M$ of dimension $n$, evaluating the Euler class of the tangent bundle on the fundamental class gives
$$\langle e(TM),[M]\rangle=\chi(M),$$
the Euler characteristic (a form of the Poincaré-Hopf theorem; see [[euler_characteristic]]). For example, the tangent bundle of the two-sphere has Euler number $2$, so it has no nowhere-vanishing section, which is the hairy ball theorem. Chern-Weil theory represents the Euler class of an even-rank bundle with a metric connection by a multiple of the Pfaffian of the curvature form. For the tangent bundle of a closed surface this recovers the Gauss-Bonnet theorem.

## Relation to Active Inference
There is no established connection. The Euler class belongs to topology, and the geometric and information-geometric treatments of Active Inference do not use characteristic classes. It appears in this knowledge base only as an item in the list of topological invariants in the hyperspatial ontology.

## Related concepts
- [[chern_classes]] — the analogous classes for complex bundles
- [[euler_characteristic]] — the number the Euler class integrates to
- [[fiber_bundles]] — the objects the class is attached to
- [[algebraic_topology]] — cohomology and the Thom isomorphism
- [[differential_geometry]] — curvature and the Pfaffian representative

## References
- Milnor, J. W., & Stasheff, J. D. (1974). Characteristic Classes (Annals of Mathematics Studies, 76). Princeton University Press.
- Bott, R., & Tu, L. W. (1982). Differential Forms in Algebraic Topology (Graduate Texts in Mathematics, 82). Springer.
- nLab authors. Euler class. nLab. https://ncatlab.org/nlab/show/Euler+class
