---
title: Self-Propelled Particles
description: Active-matter models of units that convert energy into directed motion and align with neighbours, notably the Vicsek model of flocking, and how they connect to non-equilibrium physics and Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - active_matter
  - flocking
  - non_equilibrium
semantic_relations:
  - type: relates
    links:
      - "[[bird_flocks]]"
      - "[[order_parameters]]"
      - "[[phase_transitions]]"
      - "[[non_equilibrium_thermodynamics]]"
      - "[[markov_blanket]]"
---

# Self-Propelled Particles

## Overview
Self-propelled particles are units that draw energy from their surroundings or an internal store and convert it into persistent motion, so that they are driven out of thermodynamic equilibrium at the level of each individual. Examples include bacteria, colloids driven by chemical reactions, and animals in groups. They form the basic objects of active matter physics, which asks how collective order emerges when every component propels itself.

## The Vicsek model
In the model introduced by Vicsek and colleagues in 1995, $N$ particles move at constant speed $v_0$ in a plane. At each time step a particle takes the average direction of motion of all particles within a radius $r$, including itself, and adds a random angular perturbation of amplitude $\eta$. The degree of collective motion is measured by the order parameter
$$\varphi=\frac{1}{N v_0}\left|\sum_{i=1}^{N}\mathbf{v}_i\right|,$$
which is near zero for disordered motion and near one when all particles move together. As noise decreases or density increases, the system moves from disordered to ordered motion. The nature of this transition was debated for years; later work indicated it is typically discontinuous, with dense travelling bands, although the details depend on the model variant.

## Why the model matters
Order in two dimensions with only short-range interactions is forbidden at equilibrium for systems with a continuous symmetry, yet moving particles can sustain it. This result, developed theoretically by Toner and Tu, showed that activity qualitatively changes collective behaviour. Variants add alignment with fixed numbers of neighbours, attraction and repulsion, or hydrodynamic coupling, and are used to model bird flocks, fish schools, cell migration and bacterial swarms.

## Relation to Active Inference
The connection comes through non-equilibrium physics. The free energy principle describes systems that maintain a non-equilibrium steady state separated from their environment by a Markov blanket, and active matter models are natural testbeds for such descriptions. Some papers interpret particles as minimising variational free energy with respect to beliefs about their neighbours. This is a conceptual and modelling link, still developing, and the Vicsek dynamics does not require an inference interpretation.

## Related concepts
- [[bird_flocks]] — a biological system modelled with self-propelled particles
- [[order_parameters]] — the quantity that tracks collective alignment
- [[phase_transitions]] — the shift between disordered and ordered motion
- [[non_equilibrium_thermodynamics]] — the physical setting for driven systems
- [[markov_blanket]] — the statistical boundary used in the free energy principle
- [[swarm_intelligence]] — collective problem solving by interacting agents

## References
- Vicsek, T., Czirok, A., Ben-Jacob, E., Cohen, I., & Shochet, O. (1995). Novel type of phase transition in a system of self-driven particles. Physical Review Letters, 75, 1226-1229.
- Marchetti, M. C., Joanny, J. F., Ramaswamy, S., Liverpool, T. B., Prost, J., Rao, M., & Simha, R. A. (2013). Hydrodynamics of soft active matter. Reviews of Modern Physics, 85, 1143-1189.
