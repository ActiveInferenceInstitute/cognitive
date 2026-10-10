---
title: Derived Categories
description: The category obtained from chain complexes by inverting quasi-isomorphisms, the setting for derived functors in homological algebra, with a plain statement of its absence from current Active Inference practice.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - homological_algebra
  - category_theory
  - pure_mathematics
semantic_relations:
  - type: relates
    links:
      - "[[homological_algebra]]"
      - "[[category_theory]]"
      - "[[algebraic_topology]]"
      - "[[algebraic_geometry]]"
---

# Derived Categories

## Overview
The derived category $D(\mathcal{A})$ of an abelian category $\mathcal{A}$ (modules over a ring, sheaves on a space) is built from chain complexes in $\mathcal{A}$ by formally inverting every quasi-isomorphism, a chain map inducing an isomorphism on all cohomology groups. It was introduced by Verdier, following Grothendieck's programme. The point is that two complexes with the same cohomology should be treated as equivalent for many purposes, and that left- and right-derived functors such as $\mathrm{Tor}$ and $\mathrm{Ext}$ are most naturally defined here.

## Construction
1. Start with the category $\mathrm{Ch}(\mathcal{A})$ of complexes $\cdots\to A^{n}\xrightarrow{d}A^{n+1}\to\cdots$ with $d^2=0$.
2. Pass to the homotopy category $K(\mathcal{A})$, identifying chain maps that are homotopic.
3. Localise at quasi-isomorphisms to obtain $D(\mathcal{A})$. Morphisms are represented by roofs $A\xleftarrow{\ \sim\ }A'\to B$.

Key facts:
- $D(\mathcal{A})$ is generally not abelian, but it is **triangulated**: it has a shift functor $[1]$ and distinguished triangles $A\to B\to C\to A[1]$ replacing short exact sequences.
- For a left-exact functor $F$, the right derived functor $RF:D^+(\mathcal{A})\to D^+(\mathcal{B})$ is computed by replacing an object with an injective resolution. Its cohomology gives $R^iF$.
- $\mathrm{Ext}^i(A,B)=\mathrm{Hom}_{D}(A,B[i])$.
- Variants include the bounded derived category $D^b$, and the stable $\infty$-categorical refinements that fix defects of triangulated structure.

## Role in Active Inference
There is no established use of derived categories in Active Inference or the free-energy principle. The page exists because [[homological_algebra]] and the ontology notes link to it as part of a wider mathematical taxonomy. Related work that is close but distinct: category-theoretic treatments of Bayesian inference, lenses and open dynamical systems use ordinary and monoidal categories, not derived ones, and applied topology (persistent homology) uses chain complexes and sometimes derived-category language for sheaf cohomology, but without any dependence on the free-energy principle. If a derived-category formulation of inference on graphs or sheaves appears, it would be a proposal that needs separate justification.

## Related concepts
- [[homological_algebra]] — parent subject providing complexes and exactness
- [[category_theory]] — language of localisation and triangulation
- [[algebraic_topology]] — homology and cohomology as the original examples
- [[algebraic_geometry]] — where derived categories of coherent sheaves are studied
- [[topology]] — base setting for sheaves and cohomology
- [[type_theory]] — another formal foundation sometimes combined with categorical semantics

## References
- Weibel, C. A. (1994). *An Introduction to Homological Algebra*. Cambridge University Press.
- Gelfand, S. I. and Manin, Yu. I. (2003). *Methods of Homological Algebra*, 2nd ed. Springer.
