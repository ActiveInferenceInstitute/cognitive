---
title: Euler Characteristic
description: The Euler characteristic is an integer topological invariant, computed from vertices, edges and faces or from Betti numbers, that distinguishes surfaces and links topology to curvature through the Gauss-Bonnet theorem.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - topology
  - algebraic_topology
  - differential_geometry
semantic_relations:
  - type: relates
    links:
      - "[[topology]]"
      - "[[algebraic_topology]]"
      - "[[topological_spaces]]"
      - "[[topological_data_analysis]]"
      - "[[differential_geometry]]"
---

# Euler Characteristic

## Overview
The Euler characteristic $\chi$ is an integer assigned to a topological space that does not change under homeomorphism, and more generally under homotopy equivalence. Euler first observed for convex polyhedra that the numbers of vertices $V$, edges $E$ and faces $F$ satisfy $V-E+F=2$. The characteristic generalises this count to a wide class of spaces and is one of the simplest invariants available.

## Definitions
For a finite cell complex with $c_k$ cells of dimension $k$,
$$\chi=\sum_{k\ge 0}(-1)^k c_k.$$
Homology gives an alternative formula with the same value, $\chi=\sum_{k}(-1)^k b_k$, where $b_k$ is the $k$-th Betti number, the rank of the $k$-th homology group. The equality of the two expressions shows that the cell count is independent of the chosen decomposition.

Several values are standard. A point has $\chi=1$, a circle $0$, a sphere $2$, a torus $0$, the real projective plane $1$ and the Klein bottle $0$. A closed orientable surface of genus $g$ has $\chi=2-2g$, so the characteristic determines the surface within that class.

## Properties
The characteristic is additive in the sense that $\chi(A\cup B)=\chi(A)+\chi(B)-\chi(A\cap B)$ for suitable subspaces. It is multiplicative for products, $\chi(X\times Y)=\chi(X)\chi(Y)$, and for finite covering spaces with $n$ sheets it multiplies by $n$. It is a coarse invariant: two spaces with the same $\chi$ need not be homeomorphic.

For a closed two-dimensional Riemannian manifold the Gauss-Bonnet theorem states
$$\int_M K\,dA=2\pi\,\chi(M),$$
where $K$ is the Gaussian curvature. Total curvature is thus determined by topology, even though the curvature can be distributed in many ways. The Chern-Gauss-Bonnet theorem extends this to higher even dimensions. In applied topology the Euler characteristic of a growing sequence of spaces, recorded as a curve, is a cheap summary used in the analysis of images and point-cloud data.

## Relation to Active Inference
There is no derivation connecting the Euler characteristic to the free energy principle. It is a general topological tool, and any use in this context is as part of topological data analysis of neural or behavioural data, or in mathematical discussions of the geometry of state spaces. It is listed here among the geometric and topological background and not as a component of the theory.

## Related concepts
- [[topology]] — the field in which the invariant is defined
- [[algebraic_topology]] — homology and Betti numbers
- [[topological_spaces]] — the objects to which it is assigned
- [[topological_data_analysis]] — applied use of Euler characteristic curves
- [[differential_geometry]] — Gauss-Bonnet and curvature
- [[homological_algebra]] — chain complexes, of which the alternating sum is an invariant

## References
- Hatcher, A. (2002). Algebraic Topology. Cambridge University Press.
- Munkres, J. R. (1984). Elements of Algebraic Topology. Addison-Wesley.
- do Carmo, M. P. (1976). Differential Geometry of Curves and Surfaces. Prentice-Hall.
