---
title: Gauge Theories
description: Field theories whose equations are invariant under local symmetry transformations, formulated geometrically with connections and curvature on principal bundles.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - physics
  - symmetry
  - differential_geometry
  - quantum_field_theory
aliases:
  - gauge theory
  - gauge symmetry
semantic_relations:
  - type: foundation
    links:
      - "[[differential_geometry]]"
      - "[[lie_groups]]"
      - "[[fiber_bundles]]"
  - type: relates
    links:
      - "[[quantum_field_theory]]"
      - "[[noether_theorem]]"
      - "[[holonomy]]"
      - "[[berry_connection]]"
      - "[[information_geometry]]"
---

# Gauge Theories

## Overview
A gauge theory is a physical theory whose equations are unchanged when the fields undergo a symmetry transformation that can be chosen independently at each point of spacetime. Maxwell's electrodynamics is the first example. The Stanford Encyclopedia of Philosophy entry by Dougherty notes that gauge potentials appear throughout the Standard Model of particle physics, the current best theory of the strong, weak and electromagnetic forces, and traces the gauge principle to Hermann Weyl's proposal of 1928. Philosophical debate about gauge theories mainly concerns how they should be interpreted.

## Abelian example
In electromagnetism a charged field $\psi$ can be multiplied by a phase that varies from point to point, $\psi(x)\mapsto e^{i\alpha(x)}\psi(x)$. Ordinary derivatives of $\psi$ are not preserved by this, so the derivative is replaced by a covariant derivative that contains a gauge potential $A_\mu$:

$$D_\mu=\partial_\mu - i q A_\mu, \qquad A_\mu\mapsto A_\mu+\tfrac{1}{q}\partial_\mu\alpha .$$

The field strength $F_{\mu\nu}=\partial_\mu A_\nu-\partial_\nu A_\mu$ is unchanged by the second transformation, and it is the quantity that enters the physical equations. Requiring local invariance therefore forces the existence of an interaction field.

## Non-abelian generalisation
Yang and Mills (1954) extended the construction from the phase group $U(1)$ to the non-abelian group $SU(2)$, applying it to isotopic spin. For a general compact Lie group $G$ with generators in the Lie algebra, the potential $A_\mu$ takes values in the Lie algebra and the field strength acquires a commutator term,

$$F_{\mu\nu}=\partial_\mu A_\nu-\partial_\nu A_\mu+[A_\mu,A_\nu] ,$$

absorbing the coupling constant into $A$. The gauge fields now interact with themselves. The Standard Model uses the product group $SU(3)\times SU(2)\times U(1)$.

## Geometric formulation
Wu and Yang (1975) established that gauge potentials and field strengths are the physicist's names for connections and curvature on a principal bundle, with the group $G$ as the fibre; see [[fiber_bundles]]. The phase factor obtained by transporting a charged particle around a closed loop is the holonomy of the connection (see [[holonomy]]), and the Berry phase of quantum mechanics is an instance of the same structure (see [[berry_connection]]). Because configurations related by a gauge transformation describe the same physical situation, the theory contains redundancy in its description, and the interpretive question is which structures are physically real.

## Relation to Active Inference
The connection is analogical. Information geometry and the free energy principle use differential-geometric objects such as metrics and connections on statistical manifolds, and the covariance of natural-gradient methods under reparametrisation resembles the demand that physics not depend on a choice of description. That resemblance is not a gauge symmetry in the physical sense, and no established derivation shows the free energy principle to be a gauge theory. Work that uses the language of gauge theory or bundles for beliefs should be read as a proposal for how to formalise it.

## Related concepts
- [[fiber_bundles]] — the geometry in which gauge fields are connections
- [[holonomy]] — the loop-dependent phase of a connection
- [[noether_theorem]] — how global symmetries yield conserved quantities
- [[quantum_field_theory]] — the setting of the Standard Model
- [[information_geometry]] — geometry of statistical manifolds

## References
- Dougherty, J. Gauge Theories in Physics. Stanford Encyclopedia of Philosophy (first published 28 July 2025). https://plato.stanford.edu/entries/gauge-theories/
- Yang, C. N., & Mills, R. L. (1954). Conservation of isotopic spin and isotopic gauge invariance. Physical Review, 96(1), 191-195. https://doi.org/10.1103/PhysRev.96.191
- Wu, T. T., & Yang, C. N. (1975). Concept of nonintegrable phase factors and global formulation of gauge fields. Physical Review D, 12(12), 3845-3857. https://doi.org/10.1103/PhysRevD.12.3845
