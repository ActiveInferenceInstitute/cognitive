---
title: Fiber Bundles
description: A fiber bundle is a space that locally looks like a product of a base and a fibre, with transition functions from a structure group, and underlies tangent bundles, connections, holonomy and gauge theory.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - differential_geometry
  - topology
  - gauge_theory
semantic_relations:
  - type: relates
    links:
      - "[[holonomy]]"
      - "[[berry_connection]]"
      - "[[geometric_phase]]"
      - "[[differential_geometry]]"
      - "[[topology]]"
---

# Fiber Bundles

## Overview
A fiber bundle formalises the idea of a family of spaces parametrised by another space, in a way that may twist globally. It consists of a total space $E$, a base space $B$, a surjective projection $\pi:E\to B$, and a typical fibre $F$, such that every point of $B$ has a neighbourhood $U$ with a homeomorphism $\pi^{-1}(U)\cong U\times F$ commuting with projection. Locally the bundle is a product, but globally it need not be one.

## Structure
Different local trivialisations are related on overlaps by transition functions taking values in a structure group $G$ that acts on $F$. A bundle is trivial if a single global trivialisation exists. Standard classes are:
- **Vector bundles**, with fibre $\mathbb{R}^n$ or $\mathbb{C}^n$ and structure group $GL(n)$ or a subgroup. The tangent bundle of a manifold is the main example.
- **Principal $G$-bundles**, whose fibre is the group itself acting on itself. Frame bundles and the bundles of gauge theory are of this type.

A section is a continuous choice of one point in each fibre. Vector fields are sections of the tangent bundle. The Möbius strip is a non-trivial line bundle over the circle, and the Hopf fibration $S^1\to S^3\to S^2$ is a non-trivial principal circle bundle over the two-sphere.

## Connections and holonomy
A connection specifies how to compare fibres at nearby points, that is, a rule of parallel transport along curves in $B$. Transport around a closed loop returns an element of the structure group, the holonomy, and its infinitesimal version is the curvature of the connection. A bundle with a flat connection has holonomy that depends only on the homotopy class of the loop. This covers the Levi-Civita connection of Riemannian geometry, gauge potentials in physics, and the Berry connection, whose holonomy gives the geometric phase.

## Relation to Active Inference
Bundle language enters through information geometry, where a family of probability distributions forms a manifold with tangent bundle and connections, and through geometric formulations of dynamics on state spaces. These uses are mathematical framing and not empirical claims about neural implementation. Gauge-theoretic and bundle-based formulations of the free energy principle are speculative research directions, not established results.

## Related concepts
- [[holonomy]] — the global effect of parallel transport around loops
- [[berry_connection]] — the connection on a bundle of quantum states
- [[geometric_phase]] — the physical consequence of nontrivial holonomy
- [[differential_geometry]] — the setting for tangent bundles and connections
- [[topology]] — classification of bundles by topological invariants
- [[christoffel_symbols]] — the coordinate expression of the Levi-Civita connection

## References
- Steenrod, N. (1951). The Topology of Fibre Bundles. Princeton University Press.
- Husemoller, D. (1994). Fibre Bundles (3rd ed.). Springer.
- Nakahara, M. (2003). Geometry, Topology and Physics (2nd ed.). Institute of Physics Publishing.
