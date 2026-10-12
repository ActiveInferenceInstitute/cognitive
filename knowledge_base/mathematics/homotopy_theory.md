---
title: Homotopy Theory
description: The study of continuous deformations of maps and spaces, including homotopy equivalence, the fundamental group, higher homotopy groups and fibrations, as a core tool of algebraic topology.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - topology
  - algebraic_topology
  - homotopy
semantic_relations:
  - type: relates
    links:
      - "[[topology]]"
      - "[[algebraic_topology]]"
      - "[[topological_spaces]]"
      - "[[fiber_bundles]]"
      - "[[category_theory]]"
---

# Homotopy Theory

## Overview
Homotopy theory studies spaces and continuous maps up to continuous deformation. Two maps $f,g:X\to Y$ are homotopic if there is a continuous map $H:X\times[0,1]\to Y$ with $H(\cdot,0)=f$ and $H(\cdot,1)=g$. Two spaces are homotopy equivalent if there are maps $f:X\to Y$ and $g:Y\to X$ whose composites are homotopic to the identities. Homotopy equivalence is coarser than homeomorphism: a disc, a point and a solid ball are all homotopy equivalent, while a circle is not equivalent to a point.

## Homotopy groups
The fundamental group $\pi_1(X,x_0)$ consists of homotopy classes of loops at $x_0$, with composition given by concatenation. For the circle, $\pi_1(S^1)\cong\mathbb{Z}$, the integer being the winding number. A space with trivial fundamental group that is path connected is simply connected. The higher homotopy groups $\pi_n(X,x_0)$ are defined using maps from the $n$-sphere and are abelian for $n\ge 2$. They are notoriously hard to compute: even the homotopy groups of spheres, such as $\pi_{n+k}(S^n)$, remain an open area of research.

Several structural results organise the subject. A fibration $F\to E\to B$ yields a long exact sequence $\cdots\to\pi_n(F)\to\pi_n(E)\to\pi_n(B)\to\pi_{n-1}(F)\to\cdots$. Whitehead's theorem states that a weak homotopy equivalence between CW complexes, meaning a map inducing isomorphisms on all $\pi_n$, is a homotopy equivalence. Homotopy groups are invariants distinct from homology groups, and the Hurewicz theorem relates the two in the first non-vanishing degree of a simply connected space.

## Role in Active Inference
Homotopy theory does not appear in the core derivations of Active Inference. It is relevant through related programmes: topological data analysis uses homotopy-invariant summaries such as persistent homology to describe the shape of neural and behavioural data, and homotopy type theory offers a foundation for mathematics in which paths between objects are first-class (see [[type_theory]]). Applying either to models of agents is an active but largely exploratory direction, and any link to the free energy principle is at the level of analogy and tooling rather than derivation.

## Related concepts
- [[topology]] — the broader field in which homotopy invariants are defined
- [[algebraic_topology]] — homotopy and homology groups as algebraic invariants
- [[topological_spaces]] — the objects whose maps are deformed
- [[fiber_bundles]] — bundles are examples of fibrations and give long exact sequences
- [[type_theory]] — homotopy type theory reads types as spaces
- [[topological_data_analysis]] — shape-based summaries of data

## References
- Hatcher, A. (2002). Algebraic Topology. Cambridge University Press.
- May, J. P. (1999). A Concise Course in Algebraic Topology. University of Chicago Press.
- Nakahara, M. (2003). Geometry, Topology and Physics (2nd ed.). Institute of Physics Publishing.
