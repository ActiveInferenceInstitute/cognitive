---
title: Order Parameters
description: Macroscopic variables that quantify the degree of order in a system, their role in Landau theory and synergetics, and how slow collective variables are used in Free Energy Principle arguments.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistical_physics
  - phase_transitions
  - synergetics
semantic_relations:
  - type: relates
    links:
      - "[[phase_transitions]]"
      - "[[symmetry_breaking]]"
      - "[[synergetics]]"
      - "[[critical_phenomena]]"
      - "[[renormalization_group]]"
      - "[[free_energy_principle]]"
---

# Order Parameters

## Overview
An order parameter is a macroscopic quantity that measures the degree of order in a system. It is typically zero in the disordered phase and non-zero in the ordered phase, so its behaviour signals a [[phase_transitions|phase transition]]. Examples include the magnetisation of a ferromagnet, the density difference between liquid and gas, and the complex amplitude of a superfluid condensate.

## Landau theory
Landau's phenomenological theory expands the free energy in powers of the order parameter $\eta$, keeping the terms allowed by symmetry. For a system with $\eta\to-\eta$ symmetry,
$$F(\eta)=F_0+\alpha\,(T-T_c)\,\eta^2+\beta\,\eta^4,\qquad \alpha,\beta>0.$$
Above $T_c$ the minimum is at $\eta=0$. Below $T_c$ the minima are at $\eta^2=\alpha(T_c-T)/(2\beta)$, so $\eta\propto(T_c-T)^{1/2}$ and the system chooses one of two equivalent states, which is spontaneous [[symmetry_breaking]]. This mean-field exponent is modified by fluctuations near the critical point, where the [[renormalization_group]] gives the correct critical behaviour (see [[critical_phenomena]]).

## Synergetics and slaving
Haken's synergetics applies the same idea to open, driven systems far from equilibrium, such as lasers and convection cells. Near an instability, a few slowly varying modes become unstable and act as order parameters, while the many fast, stable modes follow them adiabatically. This slaving principle reduces a high-dimensional description to a small set of collective equations (see [[synergetics]]).

## Relation to Active Inference
The link is conceptual with partial formal content. Work in the [[free_energy_principle]] tradition uses the language of slow collective variables constraining fast ones, and the separation of timescales between internal and external states mirrors the order parameter and slaving picture. Applications of synergetics to coordination dynamics in neuroscience predate and inform this usage. Whether a given cognitive variable is a true order parameter in the physical sense must be established empirically for each system. It is not implied by the principle itself.

## Related concepts
- [[phase_transitions]] — transitions in which the order parameter changes
- [[symmetry_breaking]] — the mechanism that makes it non-zero
- [[synergetics]] — order parameters and slaving in open systems
- [[critical_phenomena]] — behaviour of the order parameter near $T_c$
- [[renormalization_group]] — the theory of scale-dependent order and critical exponents

## References
- Landau, L. D., & Lifshitz, E. M. (1980). Statistical Physics, Part 1 (3rd ed.). Pergamon Press.
- Haken, H. (1983). Synergetics: An Introduction (3rd ed.). Springer.
- Chaikin, P. M., & Lubensky, T. C. (1995). Principles of Condensed Matter Physics. Cambridge University Press.
