---
title: Topological Quantum Field Theory
description: A quantum field theory whose observables depend only on the topology of spacetime, axiomatised by Atiyah as a functor from a category of cobordisms to vector spaces.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - topology
  - mathematical_physics
  - cobordism
semantic_relations:
  - type: relates
    links:
      - "[[quantum_field_theory]]"
      - "[[category_theory]]"
      - "[[monoidal_categories]]"
      - "[[functors]]"
      - "[[algebraic_topology]]"
      - "[[differential_geometry]]"
---

# Topological Quantum Field Theory

## Overview
A topological quantum field theory (TQFT) is a quantum field theory in which physical quantities do not depend on a metric on spacetime, only on its topology. Witten (1988) introduced the idea in a physical setting, relating a quantum field theory to the Donaldson invariants of four-manifolds, and Witten (1989) later connected Chern-Simons theory to the Jones polynomial of knots. Atiyah (1988) gave a mathematical definition that does not mention fields at all.

## Atiyah's axioms
Fix a dimension $n$. A TQFT $Z$ assigns:
- to each closed oriented $(n-1)$-dimensional manifold $\Sigma$ a finite-dimensional complex vector space $Z(\Sigma)$ (the "state space");
- to each compact oriented $n$-dimensional manifold $M$ with boundary, regarded as a **cobordism** from an incoming boundary $\Sigma_0$ to an outgoing boundary $\Sigma_1$, a linear map $Z(M):Z(\Sigma_0)\to Z(\Sigma_1)$.

These assignments are required to satisfy:
1. **Functoriality.** Gluing cobordisms end to end corresponds to composing linear maps, and the cylinder $\Sigma\times[0,1]$ is assigned the identity of $Z(\Sigma)$.
2. **Monoidality.** A disjoint union corresponds to a tensor product, $Z(\Sigma\sqcup\Sigma')=Z(\Sigma)\otimes Z(\Sigma')$, and the empty manifold is assigned the complex numbers, $Z(\varnothing)=\mathbb{C}$.
3. **Orientation.** Reversing the orientation of $\Sigma$ gives the dual space $Z(\Sigma)^{*}$.

In categorical language, a TQFT is a symmetric monoidal functor $Z:\mathrm{Cob}_n\to\mathrm{Vect}_{\mathbb{C}}$ (see [[functors]] and [[monoidal_categories]]). A closed $n$-manifold is a cobordism from the empty manifold to itself, so it is assigned a linear map $\mathbb{C}\to\mathbb{C}$, that is, a number $Z(M)$ which is an invariant of $M$. One consequence of the axioms is that $Z(\Sigma\times S^1)=\dim Z(\Sigma)$.

## Examples and classification
In dimension two, TQFTs correspond exactly to commutative Frobenius algebras, with the algebra being the state space of the circle (Kock, 2003). Chern-Simons theory gives a three-dimensional example, in which expectation values of Wilson loops along knots give knot invariants such as the Jones polynomial (Witten, 1989). Extended theories, which assign data also to manifolds of lower dimension, are classified by the cobordism hypothesis, whose proof was sketched by Lurie (2009).

## Relation to Active Inference
No established derivation connects TQFT to Active Inference. The resemblance is formal. TQFT assigns data to pieces of a space and composes them on gluing, and categorical treatments of inference likewise compose models (see [[category_theory]]). This is a shared style of mathematics, not a shared result, and it should not be taken to say that inference systems are topological field theories.

## Related concepts
- [[quantum_field_theory]] — the physical theories of which TQFTs are a special case
- [[category_theory]] — the language of the axioms
- [[monoidal_categories]] — the structure behind disjoint unions as tensor products
- [[functors]] — the form of the assignment $Z$
- [[algebraic_topology]] — invariants of manifolds
- [[differential_geometry]] — manifolds and their structures

## References
- Atiyah, M. (1988). Topological quantum field theory. Publications Mathématiques de l'IHÉS, 68, 175-186. https://doi.org/10.1007/BF02698547
- Witten, E. (1988). Topological quantum field theory. Communications in Mathematical Physics, 117(3), 353-386. https://doi.org/10.1007/BF01223371
- Witten, E. (1989). Quantum field theory and the Jones polynomial. Communications in Mathematical Physics, 121(3), 351-399. https://doi.org/10.1007/BF01217730
- Kock, J. (2003). Frobenius Algebras and 2D Topological Quantum Field Theories. Cambridge University Press.
- Lurie, J. (2009). On the classification of topological field theories. arXiv:0905.0465.
