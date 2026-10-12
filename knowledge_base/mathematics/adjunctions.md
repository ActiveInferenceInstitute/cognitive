---
title: Adjunctions
description: The category-theoretic notion of a pair of adjoint functors, defined by natural bijections of hom-sets or by unit and counit, with standard examples, properties, and cautious relevance to Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - category_theory
  - functors
  - universal_properties
semantic_relations:
  - type: relates
    links:
      - "[[category_theory]]"
      - "[[functors]]"
      - "[[natural_transformations]]"
      - "[[monoidal_categories]]"
      - "[[free_energy_principle]]"
---

# Adjunctions

## Overview
An adjunction is a relationship between two functors going in opposite directions between categories, expressing that one is the best possible approximation to an inverse of the other. Adjoint functors arise throughout mathematics, as Saunders Mac Lane emphasised. Free objects, products, exponentials and Galois connections are all adjoints.

## Definition
Let $F:\mathcal C\to\mathcal D$ and $G:\mathcal D\to\mathcal C$ be functors. An **adjunction** $F\dashv G$, with $F$ the left adjoint and $G$ the right adjoint, is a family of bijections
$$\mathrm{Hom}_{\mathcal D}(FC,D)\;\cong\;\mathrm{Hom}_{\mathcal C}(C,GD)$$
for objects $C$ in $\mathcal C$ and $D$ in $\mathcal D$, natural in both $C$ and $D$. An equivalent definition uses natural transformations, the **unit** $\eta:1_{\mathcal C}\Rightarrow GF$ and the **counit** $\varepsilon:FG\Rightarrow 1_{\mathcal D}$, subject to the triangle identities
$$(\varepsilon F)\circ(F\eta)=1_F,\qquad (G\varepsilon)\circ(\eta G)=1_G.$$
Given the bijection, the unit component $\eta_C$ is the element of $\mathrm{Hom}_{\mathcal C}(C,GFC)$ corresponding to $1_{FC}$, and the counit component $\varepsilon_D$ is the element of $\mathrm{Hom}_{\mathcal D}(FGD,D)$ corresponding to $1_{GD}$.

## Examples and properties
- The **free-forgetful** adjunction: the functor sending a set to the free group (or vector space, monoid) it generates is left adjoint to the forgetful functor, because a group homomorphism out of a free group is the same as a function out of the generating set.
- **Currying**: in the category of sets, the functor $-\times B$ is left adjoint to $(-)^B$, since $\mathrm{Hom}(A\times B,C)\cong\mathrm{Hom}(A,C^B)$. The same holds in any cartesian closed category.
- **Galois connections** are adjunctions between posets, viewed as categories.

Right adjoints preserve all limits that exist, and left adjoints preserve all colimits, which gives a practical test for the existence of an adjoint, made precise by the adjoint functor theorems. Adjoints are unique up to natural isomorphism. Every adjunction yields a monad $GF$ on $\mathcal C$ and a comonad $FG$ on $\mathcal D$ (Mac Lane, 1998; Awodey, 2010).

## Relation to Active Inference
Adjunctions are not part of the standard formulation of the free energy principle. They enter through the wider programme of applying category theory to Bayesian inference and to compositional models of cognition, where functors relate syntax to semantics or models to their inference procedures, and where adjoint pairs formalise relationships such as between a specification and an implementation (Fong and Spivak, 2019, give an introduction). Any use of adjoints to characterise inference or free energy minimisation is an area of exploration rather than established theory, and claims should be checked against specific papers.

## Related concepts
- [[category_theory]] — the general setting
- [[functors]] — the maps between categories that are paired in an adjunction
- [[natural_transformations]] — the unit and counit are natural transformations
- [[monoidal_categories]] — settings used for compositional models of inference
- [[free_energy_principle]] — the theory sometimes given categorical formulations

## References
- Mac Lane, S. (1998). Categories for the working mathematician (2nd ed.). Springer.
- Awodey, S. (2010). Category theory (2nd ed.). Oxford University Press.
- Fong, B., & Spivak, D. I. (2019). An invitation to applied category theory: Seven sketches in compositionality. Cambridge University Press.
