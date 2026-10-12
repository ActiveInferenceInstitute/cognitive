---
title: Monoidal Categories
description: Categories equipped with a tensor product and unit object satisfying coherence conditions, with symmetric variants, string diagrams and their role in compositional models of processes and probability.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - category_theory
  - compositionality
  - string_diagrams
semantic_relations:
  - type: relates
    links:
      - "[[category_theory]]"
      - "[[functors]]"
      - "[[natural_transformations]]"
      - "[[adjunctions]]"
      - "[[type_theory]]"
---

# Monoidal Categories

## Overview
A monoidal category is a category with a way of combining objects and morphisms in parallel. It is a category $\mathcal{C}$ together with a bifunctor $\otimes:\mathcal{C}\times\mathcal{C}\to\mathcal{C}$, a unit object $I$, and natural isomorphisms
$$\alpha_{A,B,C}:(A\otimes B)\otimes C\to A\otimes(B\otimes C),\quad \lambda_A:I\otimes A\to A,\quad \rho_A:A\otimes I\to A,$$
called the associator and the left and right unitors. These must satisfy the pentagon equation for four-fold products and the triangle equation relating $\alpha$, $\lambda$ and $\rho$. Mac Lane's coherence theorem then says that every diagram built from these isomorphisms commutes, so brackets and units can be ignored when reasoning.

## Examples and variants
- Sets with the Cartesian product and a one-point set; vector spaces with the tensor product and the ground field; and any category with finite products, where $\otimes$ is the categorical product.
- A **symmetric** monoidal category adds a natural isomorphism $\sigma_{A,B}:A\otimes B\to B\otimes A$ with $\sigma_{B,A}\circ\sigma_{A,B}=\mathrm{id}$ and compatibility with $\alpha$ (the hexagon equations). Vector spaces and sets are symmetric. Braided categories drop the involutivity condition.
- A **strict** monoidal category has identity maps for $\alpha,\lambda,\rho$. Every monoidal category is equivalent to a strict one.
- In a **cartesian** monoidal category, copying and discarding are available for every object. Dropping this requirement allows quantum systems and resource-sensitive processes.

Morphisms in a monoidal category are drawn as string diagrams, with $\otimes$ as juxtaposition and composition as vertical stacking. The graphical calculus is sound and complete for the axioms, which is why it is used for circuits, probabilistic programs and networks (Selinger, 2011).

## Role in Active Inference
Monoidal structure supplies a language for composing models: parallel composition of subsystems by $\otimes$ and sequential composition by $\circ$. Markov categories, which are symmetric monoidal categories with copy and discard maps, formalise stochastic maps and conditional independence (Fritz, 2020), and category-theoretic constructions for compositional Active Inference and cybernetic systems build on these ideas. These are active research programmes, and the theory is still developing. They do not yet replace the standard variational derivations.

## Related concepts
- [[category_theory]] — the framework in which the structure is defined
- [[functors]] — structure-preserving maps, including monoidal functors
- [[natural_transformations]] — the associator and unitors are natural isomorphisms
- [[adjunctions]] — a closed monoidal category is defined by an adjunction with $\otimes$
- [[type_theory]] — linear and dependent type systems have monoidal semantics

## References
- Mac Lane, S. (1998). Categories for the Working Mathematician (2nd ed.). Springer.
- Selinger, P. (2011). A survey of graphical languages for monoidal categories. In New Structures for Physics, Lecture Notes in Physics 813, 289-355. Springer.
- Fritz, T. (2020). A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics. Advances in Mathematics, 370, 107239.
- Fong, B., & Spivak, D. I. (2019). An Invitation to Applied Category Theory: Seven Sketches in Compositionality. Cambridge University Press.
