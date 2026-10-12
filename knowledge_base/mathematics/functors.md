---
title: Functors
description: Functors are structure-preserving maps between categories, sending objects to objects and morphisms to morphisms while respecting composition and identities, and they are central to applied category theory and categorical accounts of inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - category_theory
  - abstract_algebra
  - compositionality
semantic_relations:
  - type: relates
    links:
      - "[[category_theory]]"
      - "[[natural_transformations]]"
      - "[[monoidal_categories]]"
      - "[[sheaf_theory]]"
      - "[[free_energy_principle]]"
---

# Functors

## Overview
A functor is a map between categories. A category consists of objects, morphisms between them, an associative composition of morphisms and an identity morphism on each object. A functor $F:\mathcal{C}\to\mathcal{D}$ assigns to each object $A$ of $\mathcal{C}$ an object $F(A)$ of $\mathcal{D}$, and to each morphism $f:A\to B$ a morphism $F(f):F(A)\to F(B)$. In this way it carries structure from one mathematical setting to another.

## Definition
The assignments must satisfy two laws:
$$F(g\circ f)=F(g)\circ F(f),\qquad F(\mathrm{id}_A)=\mathrm{id}_{F(A)}.$$
A covariant functor preserves the direction of morphisms. A contravariant functor reverses it, so that $F(f):F(B)\to F(A)$ and $F(g\circ f)=F(f)\circ F(g)$. A functor is faithful if it is injective on each set of morphisms between a fixed pair of objects, full if it is surjective on them, and an equivalence if it is full, faithful and essentially surjective on objects.

## Examples
- The forgetful functor from groups to sets keeps the underlying set and forgets the operation. Its left adjoint sends a set to the free group on it.
- The power-set construction is a covariant functor on sets, using images of subsets, and a contravariant one using preimages.
- For a fixed object $A$, the functor $\mathrm{Hom}(A,-)$ sends an object $B$ to the set of morphisms from $A$ to $B$.
- The fundamental group is a functor from pointed topological spaces to groups, which is the basis for proving results such as the non-existence of a retraction of the disk onto its boundary.
- In programming, a type constructor such as a list together with a map function is a functor on types.

Functors can themselves be compared by natural transformations, which makes functors between two fixed categories into a category.

## Relation to Active Inference
Applied category theory uses functors to describe compositional models, for example from a syntactic category of open systems to a semantic category of dynamics or of probabilistic processes. Several recent proposals formulate Bayesian inference and Active Inference in categorical terms, with composition of generative models and inference procedures handled by functors. This is an active research programme concerned with how models compose. It does not change the predictions of the standard formulation, and it remains much less developed than the variational account itself.

## Related concepts
- [[category_theory]] — the framework in which functors are defined
- [[natural_transformations]] — maps between functors
- [[monoidal_categories]] — categories with a product, used for compositional systems
- [[sheaf_theory]] — sheaves are functors satisfying a gluing condition
- [[derived_categories]] — constructions built from functors on chain complexes
- [[free_energy_principle]] — a setting for categorical formulations of inference

## References
- Mac Lane, S. (1998). Categories for the Working Mathematician (2nd ed.). Springer.
- Awodey, S. (2010). Category Theory (2nd ed.). Oxford University Press.
- Fong, B., & Spivak, D. I. (2019). An Invitation to Applied Category Theory: Seven Sketches in Compositionality. Cambridge University Press.
