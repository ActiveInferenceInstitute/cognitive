---
title: Quantum Mechanics
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - quantum_mechanics
  - hilbert_space
  - born_rule
semantic_relations:
  - type: foundation
    links:
      - "[[linear_algebra]]"
      - "[[functional_analysis]]"
  - type: relates
    links:
      - "[[quantum_information]]"
      - "[[quantum_cognition]]"
      - "[[langevin_dynamics]]"
---

# Quantum Mechanics

## Overview
Quantum mechanics is the theory of physical systems whose states are vectors in a complex Hilbert space, whose observables are Hermitian operators, and whose measurement outcomes are probabilistic. It is linear in the state but yields interference because amplitudes, not probabilities, add.

## Definition / Formalism
The standard postulates:

1. A pure state is a unit vector $|\psi\rangle$ in a Hilbert space $\mathcal H$ (up to global phase).
2. Observables are Hermitian operators; measuring $A=\sum_i a_i |i\rangle\langle i|$ yields $a_i$ with probability $p_i=|\langle i|\psi\rangle|^2$ (Born rule), after which the state is projected onto $|i\rangle$.
3. Closed systems evolve unitarily: $i\hbar\,\partial_t|\psi\rangle=H|\psi\rangle$, so $|\psi(t)\rangle=e^{-iHt/\hbar}|\psi(0)\rangle$.
4. Composite systems live in the tensor product $\mathcal H_A\otimes\mathcal H_B$; states not expressible as a product are entangled.

Worked check (two-level system): with $H=\tfrac{\hbar\omega}{2}\sigma_x$ and initial state $|0\rangle$, the probability of finding $|1\rangle$ at time $t$ is $\sin^2(\omega t/2)$ (Rabi oscillation); a numerical matrix exponential reproduces this exactly. Mixed states are density operators $\rho\succeq0$ with $\operatorname{Tr}\rho=1$ and $p_i=\langle i|\rho|i\rangle$. The uncertainty relation $\sigma_A\sigma_B\ge\tfrac12|\langle[A,B]\rangle|$ follows from the operator algebra.

## Role in Active Inference
The standard free energy principle is formulated for classical (stochastic) dynamics, and its core results do not require quantum mechanics. Three connections exist and should be graded separately. (1) A solid mathematical analogy: the Fokker-Planck operator for a gradient-drift diffusion is similar to a Schrodinger-type Hamiltonian in imaginary time, which is why path-integral methods transfer between the two. (2) Quantum probability has been proposed for human judgment and decision effects (interference, order effects); that is the topic of the quantum cognition page and remains debated. (3) Fields, Friston, Glazebrook and Levin (2022) have developed a version of the free energy principle for generic quantum systems using a formal Markov-blanket construction; this is a theoretical research programme, not an empirically established result. Nothing in this repository's agents uses quantum mechanics.

## Related concepts
- [[quantum_information]] — entropy, entanglement and channels built on the density-operator formalism
- [[quantum_field_theory]] — the relativistic, many-degree-of-freedom extension
- [[quantum_cognition]] — proposals to model cognition with quantum probability
- [[quantum_systems]] — research-level overview in this vault
- [[linear_algebra]] — the mathematical language of states and operators
- [[fokker_planck]] — classical diffusion equation with a formal quantum analogue
