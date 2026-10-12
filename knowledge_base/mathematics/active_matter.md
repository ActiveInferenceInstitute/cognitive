---
title: Active Matter
description: Systems whose constituents consume energy to move or exert forces, including flocks and bacterial suspensions, with the Vicsek model, collective phenomena, and links to non-equilibrium physics and the free energy principle.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - non_equilibrium_physics
  - collective_behavior
  - statistical_physics
semantic_relations:
  - type: relates
    links:
      - "[[non_equilibrium_thermodynamics]]"
      - "[[self_propelled_particles]]"
      - "[[self_organization]]"
      - "[[entropy_production]]"
      - "[[free_energy_principle]]"
wikidata: Q4677565
---

# Active Matter

## Overview
Active matter is the physics of systems made of many units that each convert stored or ambient energy into directed motion or mechanical force. Examples include bird flocks and fish schools, swimming bacteria and sperm, cytoskeletal filaments driven by molecular motors, and synthetic self-propelled colloids. Because energy is injected and dissipated at the scale of the individual unit, these systems are out of thermodynamic equilibrium even without an external drive on the whole, and the detailed balance that underlies equilibrium statistical mechanics is broken (Ramaswamy, 2010; Marchetti et al., 2013).

## Models and phenomena
A minimal model is the Vicsek model, in which $N$ point particles move at constant speed $v_0$ and at each time step align their direction with the average direction of neighbours within a fixed radius, subject to noise:
$$\theta_i(t+1)=\langle\theta_j(t)\rangle_{|\mathbf r_j-\mathbf r_i|<R}+\eta_i(t),\qquad \mathbf r_i(t+1)=\mathbf r_i(t)+v_0\,\mathbf e(\theta_i(t+1)),$$
where $\mathbf e(\theta)$ is the unit vector at angle $\theta$ and $\eta_i$ is random noise drawn from a bounded interval. Vicsek and colleagues (1995) found a transition from disordered motion to collective motion as noise decreases or density increases. Toner and Tu (1995) showed with a continuum theory that long-range orientational order is possible in two dimensions for such flocks, in contrast to equilibrium systems with continuous symmetry, where the Mermin-Wagner theorem forbids it.

Other characteristic phenomena include giant number fluctuations, in which fluctuations in the number of particles in a subregion grow faster than the square root of the mean, motility-induced phase separation of self-propelled particles that slow in crowds, and spontaneous flows in active fluids built from filaments and motors. Theoretical tools include hydrodynamic theory, active Langevin dynamics (see [[langevin_dynamics]]) and entropy production as a measure of irreversibility (see [[entropy_production]]).

## Relation to Active Inference
The free energy principle describes self-organising systems that maintain a Markov blanket separating internal and external states, and it is formulated for non-equilibrium steady states (see [[markov_blanket]]). Active matter supplies physical examples of such systems, and some authors discuss whether collective behaviour in flocks or swarms can be described as inference at the level of the group. These proposals are at an exploratory stage. The principle's assumptions, such as a nonequilibrium steady state with a well-defined blanket, hold only for particular models of active matter, and detailed correspondence for the Vicsek class has not been established.

## Related concepts
- [[non_equilibrium_thermodynamics]] — the general framework for driven systems
- [[self_propelled_particles]] — the basic constituents of many active systems
- [[self_organization]] — emergence of collective order
- [[entropy_production]] — a measure of irreversibility in active systems
- [[markov_blanket]] — boundary concept used in the free energy principle
- [[swarm_intelligence]] — collective behaviour in biological and artificial swarms

## References
- Vicsek, T., Czirók, A., Ben-Jacob, E., Cohen, I., & Shochet, O. (1995). Novel type of phase transition in a system of self-driven particles. Physical Review Letters, 75, 1226-1229.
- Toner, J., & Tu, Y. (1995). Long-range order in a two-dimensional dynamical XY model: How birds fly together. Physical Review Letters, 75, 4326-4329.
- Ramaswamy, S. (2010). The mechanics and statistics of active matter. Annual Review of Condensed Matter Physics, 1, 323-345.
- Marchetti, M. C., Joanny, J. F., Ramaswamy, S., Liverpool, T. B., Prost, J., Rao, M., & Simha, R. A. (2013). Hydrodynamics of soft active matter. Reviews of Modern Physics, 85, 1143-1189.
