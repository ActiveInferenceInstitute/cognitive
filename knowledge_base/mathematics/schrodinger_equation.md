---
title: Schrödinger Equation
description: The fundamental equation of non-relativistic quantum mechanics governing the evolution of a wave function, its time-independent form, variational characterisation and formal parallels with the Fokker-Planck equation.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - quantum_mechanics
  - wave_function
  - partial_differential_equations
semantic_relations:
  - type: relates
    links:
      - "[[quantum_mechanics]]"
      - "[[variational_calculus]]"
      - "[[hilbert_spaces]]"
      - "[[fokker_planck]]"
      - "[[hamiltonian_mechanics]]"
      - "[[quantum_free_energy]]"
wikidata: Q165498
---

# Schrödinger Equation

## Overview
The Schrödinger equation describes how the quantum state of a non-relativistic system changes in time. The state is a wave function $\psi(x,t)$, a complex-valued element of a Hilbert space, and its evolution is governed by
$$i\hbar\,\partial_t\psi=\hat H\psi,\qquad \hat H=-\frac{\hbar^2}{2m}\nabla^2+V(x),$$
where $\hat H$ is the Hamiltonian operator, $m$ the mass and $V$ the potential. Erwin Schrödinger published it in 1926. By the Born rule, $|\psi(x,t)|^2$ is the probability density of finding the particle at $x$.

## Properties
The equation is linear, so superpositions of solutions are solutions, and it generates unitary evolution that preserves the norm of $\psi$, hence total probability. For a time-independent potential, separation of variables gives $\psi(x,t)=\phi(x)e^{-iEt/\hbar}$, where $\phi$ satisfies the time-independent equation
$$\hat H\phi=E\phi.$$
This is an eigenvalue problem. Bound states have discrete energies $E_n$, which explains the line spectra of atoms and the quantisation of energy in the harmonic oscillator, $E_n=\hbar\omega(n+\tfrac12)$.

## Variational characterisation
The ground state minimises the energy functional $\langle\psi|\hat H|\psi\rangle/\langle\psi|\psi\rangle$ over normalised states, and excited states are stationary points subject to orthogonality constraints. This Rayleigh-Ritz principle is the basis of practical approximation methods and ties the equation to the calculus of variations. The Schrödinger equation is also the Euler-Lagrange equation of a suitable action functional, and it is related to Feynman's path integral formulation, which expresses the propagator as a sum over trajectories.

## Relation to Active Inference
The connection is mathematical and partly speculative. In the free energy principle, densities over states evolve under the Fokker-Planck equation, and for gradient-type drift this operator can be transformed into a Schrödinger-type operator in imaginary time, which makes techniques from quantum mechanics available. Separately, a small body of work formulates the free energy principle for quantum systems and examines Markov blankets in quantum settings. These are active but specialised research lines, and Active Inference as applied to neuroscience and behaviour does not require quantum mechanics.

## Related concepts
- [[quantum_mechanics]] — the theory built on this equation
- [[hilbert_spaces]] — the state space of wave functions
- [[variational_calculus]] — the mathematics behind the energy principle
- [[fokker_planck]] — a diffusion equation with a formal operator parallel
- [[quantum_free_energy]] — quantum extensions of free energy ideas

## References
- Schrödinger, E. (1926). Quantisierung als Eigenwertproblem. Annalen der Physik, 79, 361-376.
- Griffiths, D. J., & Schroeter, D. F. (2018). Introduction to Quantum Mechanics (3rd ed.). Cambridge University Press.
- Sakurai, J. J., & Napolitano, J. (2017). Modern Quantum Mechanics (2nd ed.). Cambridge University Press.
