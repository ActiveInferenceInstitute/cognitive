---
title: Natural Transformations
description: Maps between functors that assign to each object a morphism commuting with the functors' action on arrows, with the standard examples, natural isomorphism, functor categories and the Yoneda lemma.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - category_theory
  - functors
  - naturality
semantic_relations:
  - type: relates
    links:
      - "[[category_theory]]"
      - "[[functors]]"
      - "[[adjunctions]]"
      - "[[monoidal_categories]]"
      - "[[sheaf_theory]]"
---

# Natural Transformations

## Overview
A natural transformation compares two functors. Given categories $\mathcal{C},\mathcal{D}$ and functors $F,G:\mathcal{C}\to\mathcal{D}$, a natural transformation $\eta:F\Rightarrow G$ assigns to every object $X$ of $\mathcal{C}$ a morphism $\eta_X:F(X)\to G(X)$ in $\mathcal{D}$, called a component, such that for every morphism $f:X\to Y$ in $\mathcal{C}$,
$$G(f)\circ\eta_X=\eta_Y\circ F(f).$$
This commuting square is the naturality condition: the family of components is defined uniformly, without making an arbitrary choice for each object. Eilenberg and Mac Lane introduced categories, functors and natural transformations in 1945 in order to make precise the informal notion of a "natural" isomorphism.

## Examples and structure
- **Double dual.** For finite-dimensional vector spaces over a field, the evaluation map $V\to V^{**}$ is a natural isomorphism from the identity functor to the double-dual functor. An isomorphism $V\cong V^{*}$ requires a choice of basis or inner product and is not natural.
- **Natural isomorphism.** A natural transformation whose components are all isomorphisms. Two categories are equivalent when functors between them are inverse up to natural isomorphism, which is the usual correct notion of sameness for categories.
- **Functor categories.** The functors $\mathcal{C}\to\mathcal{D}$ with natural transformations as morphisms form a category $\mathcal{D}^{\mathcal{C}}$. Vertical composition of transformations is componentwise, and horizontal composition combines transformations between functors that compose.
- **Yoneda lemma.** For a locally small category $\mathcal{C}$, the natural transformations from the hom-functor $\mathrm{Hom}(X,-)$ to a set-valued functor $F$ correspond bijectively to elements of $F(X)$, naturally in $X$ and $F$. It implies that an object is determined up to isomorphism by the morphisms into it or out of it.
- **Adjunctions** can be defined by a unit and counit, which are natural transformations satisfying the triangle identities.

## Role in Active Inference
Natural transformations are not used in the standard derivation of Active Inference. They appear in categorical formulations that treat generative models as functors from a category of structures to probabilistic maps, where a natural transformation expresses a systematic translation between two such models. These constructions belong to an active research programme in applied category theory and are best regarded as a compositional reformulation of existing material, not as new predictions.

## Related concepts
- [[category_theory]] — the framework in which naturality is defined
- [[functors]] — the objects that natural transformations relate
- [[adjunctions]] — defined by unit and counit transformations
- [[monoidal_categories]] — associators and unitors are natural isomorphisms
- [[sheaf_theory]] — sheaves are functors, with morphisms as natural transformations

## References
- Eilenberg, S., & Mac Lane, S. (1945). General theory of natural equivalences. Transactions of the American Mathematical Society, 58, 231-294.
- Mac Lane, S. (1998). Categories for the Working Mathematician (2nd ed.). Springer.
- Riehl, E. (2016). Category Theory in Context. Dover Publications.
