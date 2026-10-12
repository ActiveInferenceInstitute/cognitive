---
title: Canonical Coordinates
description: Canonical coordinates are pairs of generalised positions and conjugate momenta in Hamiltonian mechanics, obeying Hamilton's equations and fixed Poisson brackets that define phase-space structure.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - hamiltonian_mechanics
  - phase_space
  - classical_mechanics
semantic_relations:
  - type: relates
    links:
      - "[[hamiltonian_mechanics]]"
      - "[[lagrangian_mechanics]]"
      - "[[canonical_transformations]]"
      - "[[symplectic_geometry]]"
      - "[[path_integral_free_energy]]"
---

# Canonical Coordinates

## Overview
In Hamiltonian mechanics a system with $n$ degrees of freedom is described by $2n$ canonical coordinates: generalised coordinates $q_i$ and their conjugate momenta $p_i$. Together they label a point in phase space. The word "canonical" indicates that the pairs are linked in the specific way that makes Hamilton's equations take their standard form.

## Definition
Given a Lagrangian $L(q,\dot q,t)$, the momentum conjugate to $q_i$ is
$$p_i=\frac{\partial L}{\partial \dot q_i}.$$
For a particle with $L=\tfrac12 m\dot q^2-V(q)$ this gives $p=m\dot q$, the ordinary momentum, but in general conjugate momenta can differ from mass times velocity. For a charged particle in a magnetic field, for example, $p=m\dot q+qA$ includes the vector potential. When the Legendre transform is regular, the Hamiltonian is $H(q,p,t)=\sum_i p_i\dot q_i-L$ and the dynamics are
$$\dot q_i=\frac{\partial H}{\partial p_i},\qquad \dot p_i=-\frac{\partial H}{\partial q_i}.$$

## Structure
The defining algebraic property is the set of fundamental Poisson brackets $\{q_i,q_j\}=0$, $\{p_i,p_j\}=0$ and $\{q_i,p_j\}=\delta_{ij}$. Equivalently, the phase space carries the symplectic two-form $\omega=\sum_i dq_i\wedge dp_i$ (up to a sign convention), and canonical coordinates are Darboux coordinates in which $\omega$ takes this constant form. Coordinates with this property are not unique; the maps between admissible sets are the [[canonical_transformations]].

A coordinate $q_k$ that does not appear in $H$ is called cyclic, and its conjugate momentum $p_k$ is then conserved, which is the simplest form of Noether's theorem for these systems.

The term should not be confused with the generalised coordinates of Friston's formulations, which are successive time derivatives of a variable rather than conjugate pairs.

## Relation to Active Inference
Canonical coordinates enter the vault through path-integral and Hamiltonian formulations of the free energy principle, where the action and momentum conjugate to a state variable appear in the dynamics of beliefs. The connection is formal: it applies the standard Hamiltonian machinery to a variational problem, and does not claim that neural or biological states are physical positions and momenta.

## Related concepts
- [[hamiltonian_mechanics]] — the theory in which canonical pairs are defined
- [[lagrangian_mechanics]] — the source of the conjugate-momentum definition
- [[canonical_transformations]] — changes of coordinates preserving the canonical form
- [[symplectic_geometry]] — the geometric structure behind the Poisson bracket
- [[generalized_coordinates]] — the different, derivative-based use of the term in Active Inference
- [[noether_theorem]] — conserved momenta from cyclic coordinates

## References
- Goldstein, H., Poole, C., & Safko, J. (2002). Classical Mechanics (3rd ed.). Addison-Wesley.
- Arnold, V. I. (1989). Mathematical Methods of Classical Mechanics (2nd ed.). Springer.
