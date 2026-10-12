---
title: Brownian Ratchets
description: Brownian ratchets are models of directed transport from thermal fluctuations, explaining why equilibrium noise cannot be rectified and how nonequilibrium driving lets molecular motors do work.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistical_physics
  - molecular_motors
  - nonequilibrium
semantic_relations:
  - type: relates
    links:
      - "[[brownian_motion]]"
      - "[[molecular_motors]]"
      - "[[maxwell_demon]]"
      - "[[fluctuation_dissipation]]"
      - "[[non_equilibrium_thermodynamics]]"
---

# Brownian Ratchets

## Overview
A Brownian ratchet is a model in which a particle subject to thermal noise and an asymmetric, periodic potential moves with a net drift. The idea is that an asymmetric structure might rectify random fluctuations into directed motion. Analysis of the model shows that this is impossible at thermal equilibrium and possible when the system is driven away from it, which makes ratchets a standard framework for molecular motors.

## The equilibrium paradox
The classic example is the ratchet and pawl discussed by Feynman. A ratchet wheel with asymmetric teeth is coupled by an axle to vanes immersed in a gas, and a pawl allows rotation in one direction only. If both parts are at the same temperature, the pawl itself fluctuates and occasionally lets the wheel slip backwards, and the net rotation averages to zero. If the pawl is colder than the vanes, net rotation can occur, but this uses a temperature difference and acts as a heat engine. In general, detailed balance forbids net current at a single temperature, in agreement with the second law.

## Nonequilibrium ratchets
Net transport arises when detailed balance is broken. Two common mechanisms are:
- **Flashing ratchets.** The asymmetric potential is switched on and off, so particles spread by diffusion during the off phase and are then captured when it is on again. Because the wells are asymmetric, the spread cloud is more likely to be captured by the well on one particular side.
- **Rocking ratchets.** The potential is tilted by a periodic or noisy force with zero mean, and the asymmetry makes the response to forward and backward tilts unequal.

In both cases energy is supplied by the driving, such as chemical free energy from ATP hydrolysis in biological motors, and the motion is described by a Fokker-Planck equation with a periodic potential. Directed motion therefore costs dissipation.

## Relation to Active Inference
The link is an analogy and a shared physical background. Both the ratchet literature and the free energy principle concern systems that maintain non-equilibrium steady states by dissipating free energy, and discussions of Maxwell's demon and information engines connect ratchets to the thermodynamics of measurement and feedback. Brownian ratchets are not part of the formal derivation of Active Inference.

## Related concepts
- [[brownian_motion]] — the fluctuation source that ratchets exploit
- [[molecular_motors]] — biological realisations of ratchet-like transport
- [[maxwell_demon]] — the information-based version of the rectification question
- [[fluctuation_dissipation]] — the equilibrium relation between noise and friction
- [[non_equilibrium_thermodynamics]] — the setting in which net transport is allowed
- [[fokker_planck]] — the equation governing the probability density

## References
- Feynman, R. P., Leighton, R. B., & Sands, M. (1963). The Feynman Lectures on Physics, Volume I. Addison-Wesley. (Chapter 46, Ratchet and pawl.)
- Astumian, R. D. (1997). Thermodynamics and kinetics of a Brownian motor. Science, 276, 917-922.
- Reimann, P. (2002). Brownian motors: Noisy transport far from equilibrium. Physics Reports, 361, 57-265.
