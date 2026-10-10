---
title: Integrable Systems
description: Dynamical systems with enough independent conserved quantities to be solved exactly, from Liouville-Arnold integrability and action-angle variables to soliton equations, and their absence from standard Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - hamiltonian_mechanics
  - symplectic_geometry
  - pure_mathematics
semantic_relations:
  - type: relates
    links:
      - "[[hamiltonian_mechanics]]"
      - "[[poisson_geometry]]"
      - "[[symplectic_geometry]]"
      - "[[conservation_laws]]"
---

# Integrable Systems

## Overview
An integrable system is a dynamical system whose motion can be determined by integrals (quadratures) because it has many conserved quantities. Such systems are exceptional: most nonlinear systems are non-integrable and often chaotic. Integrable examples include the Kepler problem, the harmonic oscillator, Euler's top, the Toda lattice, and soliton equations such as Korteweg-de Vries.

## Definitions
**Liouville-Arnold.** A Hamiltonian system with $n$ degrees of freedom on a $2n$-dimensional symplectic manifold is integrable if it has $n$ functionally independent first integrals $F_1=H,\dots,F_n$ in involution, $\{F_i,F_j\}=0$ (see [[poisson_geometry]]). For compact connected common level sets the Liouville-Arnold theorem says they are $n$-tori, and one can choose action-angle coordinates $(I,\theta)$ in which
$$\dot I_i=0,\qquad \dot\theta_i=\omega_i(I).$$
Motion is then quasi-periodic on each torus, with the torus filled densely when the frequencies are rationally independent. Perturbations of integrable systems are described by KAM theory: most tori survive small perturbation, while resonant ones break up and give rise to chaos ([[chaos_theory]]).

**Infinite-dimensional case.** For field equations such as KdV, integrability is signalled by a Lax pair $\dot L=[M,L]$, which makes the spectrum of $L$ time-independent, giving infinitely many conserved quantities and a solution method by inverse scattering.

**Related notions.** Superintegrability (more than $n$ independent integrals), Painleve property and Hamiltonian reduction by symmetry.

## Role in Active Inference
Integrable systems have no established role in Active Inference. The indirect connection is structural. The free-energy principle's decomposition of flow into gradient and solenoidal parts has a Hamiltonian-like solenoidal component, and conservation laws on the level sets of surprise make that component resemble conservative dynamics ([[conservation_laws]]). Actual generative models and agents are dissipative, stochastic and usually non-integrable, so the Liouville-Arnold machinery does not apply directly. Integrable models could serve as solvable test cases for inference algorithms, such as linear oscillators in Kalman-filter checks, which is a methodological use only.

## Related concepts
- [[hamiltonian_mechanics]] — framework where integrability is defined
- [[symplectic_geometry]] — geometric structure underlying action-angle coordinates
- [[poisson_geometry]] — Poisson brackets and involutive integrals
- [[conservation_laws]] — the conserved quantities on which integrability rests
- [[chaos_theory]] — non-integrable behaviour after perturbation
- [[geometric_mechanics]] — wider setting for reduction and symmetry

## References
- Arnold, V. I. (1989). *Mathematical Methods of Classical Mechanics*, 2nd ed. Springer.
