---
title: Black Hole Thermodynamics
description: The correspondence between the laws of black hole mechanics and the laws of thermodynamics, including the Bekenstein-Hawking entropy and Hawking temperature, and its use in speculative links to the free energy principle.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - thermodynamics
  - general_relativity
  - entropy
semantic_relations:
  - type: relates
    links:
      - "[[holographic_principle]]"
      - "[[entropy]]"
      - "[[thermodynamics]]"
      - "[[information_thermodynamics]]"
      - "[[quantum_free_energy]]"
      - "[[free_energy_principle]]"
wikidata: Q911284
---

# Black Hole Thermodynamics

## Overview
Black hole thermodynamics is the body of results showing that black holes obey laws formally identical to those of thermodynamics, and, once quantum effects are included, are in fact thermodynamic objects with an entropy and a temperature. It arose in the early 1970s from the observation that the area of an event horizon behaves like entropy: in classical general relativity it never decreases.

## The laws
In the work of Bardeen, Carter and Hawking (1973), the laws of black hole mechanics for stationary black holes read, with $\kappa$ the surface gravity, $A$ the horizon area, $M$ the mass, $J$ the angular momentum, $Q$ the charge and $\Omega$, $\Phi$ the horizon angular velocity and electric potential:
- **Zeroth law**: $\kappa$ is constant over the horizon, like temperature in equilibrium.
- **First law**: $dM=\dfrac{\kappa}{8\pi G}\,dA+\Omega\,dJ+\Phi\,dQ$ in units where $c=1$, which parallels $dE=T\,dS+\dots$
- **Second law**: in classical general relativity, with matter obeying the null energy condition, the horizon area never decreases, $dA\ge 0$.
- **Third law**: $\kappa=0$ cannot be reached by a finite sequence of operations.

Bekenstein (1973) proposed that the area is proportional to a genuine entropy. Hawking (1975) showed that black holes emit radiation with a thermal spectrum at temperature $T_H=\hbar\kappa/(2\pi c\,k_B)$, which fixes the constant:
$$S_{BH}=\frac{k_B c^3 A}{4G\hbar}.$$
For a Schwarzschild black hole of mass $M$, $T_H=\hbar c^3/(8\pi G M k_B)$, so the temperature decreases as the mass increases, and the black hole has negative heat capacity. Because radiation lowers the area, the classical area theorem is replaced by the **generalised second law**: the sum of black hole entropy and the entropy of matter outside the horizon does not decrease.

The entropy scales with area rather than volume, which suggested the holographic principle (see [[holographic_principle]]). The information loss problem raised by evaporation is still debated.

## Relation to Active Inference
The connection is speculative. Some proposals formulate the free energy principle for quantum systems using holographic screens as the boundary playing the role of a Markov blanket, motivated by the area-law bound on information (see [[quantum_free_energy]] and [[markov_blanket]]). These are theoretical extensions at an exploratory stage, and standard Active Inference models of perception and action do not depend on black hole physics. The thermodynamic link between entropy, information and free energy is the more established common ground (see [[information_thermodynamics]]).

## Related concepts
- [[holographic_principle]] — bound on information set by boundary area
- [[entropy]] — the quantity assigned to the horizon
- [[thermodynamics]] — the laws that the mechanics mirrors
- [[information_thermodynamics]] — links between information and physical entropy
- [[quantum_free_energy]] — quantum extensions of free energy formulations
- [[markov_blanket]] — boundary concept compared with holographic screens

## References
- Bekenstein, J. D. (1973). Black holes and entropy. Physical Review D, 7, 2333-2346.
- Bardeen, J. M., Carter, B., & Hawking, S. W. (1973). The four laws of black hole mechanics. Communications in Mathematical Physics, 31, 161-170.
- Hawking, S. W. (1975). Particle creation by black holes. Communications in Mathematical Physics, 43, 199-220.
