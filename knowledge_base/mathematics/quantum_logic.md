---
title: Quantum Logic and Quantum Probability
description: The orthocomplemented lattice of closed subspaces of a Hilbert space, proposed by Birkhoff and von Neumann as the logic of quantum propositions, and the non-Boolean generalisation of probability theory built on it.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - quantum_mechanics
  - logic
  - probability
  - lattice_theory
aliases:
  - quantum probability
  - quantum probability theory
  - Birkhoff-von Neumann quantum logic
semantic_relations:
  - type: relates
    links:
      - "[[quantum_mechanics]]"
      - "[[hilbert_spaces]]"
      - "[[probability_theory]]"
      - "[[measurement_problem]]"
      - "[[quantum_cognition]]"
      - "[[logic]]"
      - "[[quantum_free_energy]]"
---

# Quantum Logic and Quantum Probability

## Overview
In quantum mechanics a physical system is associated with a Hilbert space $\mathbf{H}$, whose unit vectors represent states, and an observable is a self-adjoint operator on $\mathbf{H}$. Yes-no observables are projection operators ($P^2=P$), which correspond one-to-one with the closed subspaces of $\mathbf{H}$. Birkhoff and von Neumann (1936) proposed treating these projections as propositions about the system and studying the algebraic structure they form as a "logic". Alexander Wilce's Stanford Encyclopedia of Philosophy entry describes the formal apparatus of quantum mechanics as a generalisation of classical probability in which the Boolean algebra of events is replaced by this quantum logic of projections.

## The lattice of projections
Ordered by inclusion, the closed subspaces of $\mathbf{H}$ form a complete lattice, where the meet of subspaces is their intersection and the join is the closed span of their union. The map $\mathbf{M}\mapsto\mathbf{M}^{\perp}$ (the orthogonal complement) makes the lattice orthocomplemented. It is not distributive, because a typical closed subspace has infinitely many complements. Two projections $P$ and $Q$ commute if and only if the sublattice they generate is Boolean, and the entry draws the conclusion that commuting observables, which are simultaneously measurable, can be treated by classical logic. When $PQ=QP$ the meet is $P\wedge Q=PQ$ and the join is $P\vee Q=P+Q-PQ$.

## Quantum probability
A probability measure on the lattice $L(\mathbf{H})$ is a map $\mu:L\to[0,1]$ with $\mu(\mathbf{1})=1$ that is additive over orthogonal projections. A unit vector $u$ gives the measure $\mu_u(P)=\langle Pu,u\rangle$, and mixtures of such measures correspond to density operators $W$, with $\mu(P)=\operatorname{Tr}(WP)$.

**Gleason's theorem** (1957): if $\dim\mathbf{H}>2$, every countably additive probability measure on $L(\mathbf{H})$ has this form. A consequence is that no such measure takes only the values 0 and 1, which is often taken to rule out simple non-contextual hidden variables. Wilce's entry goes on to cover generalised probability theory, test spaces and effect algebras, and the treatment of composite systems.

## Interpretations
The entry separates a realist reading of quantum logic, in which the lattice is the logic of properties of a system, from an operational reading based on the structure of experimental tests. Whether the lattice is a revision of classical logic or a formal device for organising quantum probabilities is a matter on which the interpretations differ.

## Relation to Active Inference
Active Inference as usually formulated uses classical probability. Quantum probability appears in cognitive science in "quantum cognition" models of judgment, where the non-commutation of events is used for order effects, without any claim that the brain is quantum-mechanical (see [[quantum_cognition]]). The quantum-mechanical analogue of free energy is treated separately (see [[quantum_free_energy]]). These are separate research programmes and not part of the standard formalism.

## References
- Wilce, A. (2002). Quantum Logic and Probability Theory. In E. N. Zalta & U. Nodelman (Eds.), The Stanford Encyclopedia of Philosophy (first published 4 February 2002, substantive revision 10 August 2021). https://plato.stanford.edu/entries/qt-quantlog/
- Birkhoff, G., & von Neumann, J. (1936). The logic of quantum mechanics. Annals of Mathematics, 37(4), 823-843. doi:10.2307/1968621
- Gleason, A. M. (1957). Measures on the closed subspaces of a Hilbert space. Journal of Mathematics and Mechanics (now Indiana University Mathematics Journal), 6(4), 885-893. doi:10.1512/iumj.1957.6.56050
