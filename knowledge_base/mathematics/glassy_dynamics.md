---
title: Glassy Dynamics
description: Glassy dynamics are the very slow, history-dependent relaxation of disordered systems with rugged energy landscapes, seen in supercooled liquids and spin glasses and marked by aging and broken ergodicity.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistical_physics
  - disordered_systems
  - nonequilibrium
semantic_relations:
  - type: relates
    links:
      - "[[broken_ergodicity]]"
      - "[[free_energy_landscape]]"
      - "[[attractor_dynamics]]"
      - "[[non_equilibrium_thermodynamics]]"
      - "[[statistical_physics]]"
---

# Glassy Dynamics

## Overview
Glassy dynamics describes systems whose relaxation to equilibrium becomes so slow that on experimental timescales they remain out of equilibrium. Typical examples are supercooled liquids approaching the glass transition, structural and polymer glasses, and spin glasses, which are magnets with frustrated random interactions. The common features are a rapid growth of relaxation times on cooling, non-exponential relaxation, and dependence on the history of the sample.

## Characteristic phenomena
- **Slow relaxation.** In supercooled liquids the structural relaxation time grows by many orders of magnitude over a modest temperature range. A frequently used empirical description is the Vogel-Fulcher-Tammann law, $\tau=\tau_0\exp\!\big[DT_0/(T-T_0)\big]$, which diverges at a finite temperature $T_0$. Whether a true divergence exists is debated, and an Arrhenius law $\tau\propto e^{E/k_BT}$ describes "strong" glass formers.
- **Stretched relaxation.** Correlation functions often decay as a stretched exponential, $\exp[-(t/\tau)^{\beta}]$ with $0<\beta<1$, interpreted as a distribution of relaxation times.
- **Aging.** Two-time correlation functions depend on the waiting time $t_w$ since the quench, not only on the time difference, so the system's response changes as it ages. This marks the breakdown of time-translation invariance and of the fluctuation-dissipation relation.
- **Rugged landscapes.** The potential or free-energy landscape has many metastable states separated by barriers, and the system gets trapped, which gives [[broken_ergodicity]].

## Theoretical pictures
For mean-field spin glasses, Parisi's solution describes a hierarchy of states by replica symmetry breaking. For structural glasses, competing accounts include mode-coupling theory, random first-order transition theory, kinetically constrained models and dynamical facilitation. Which description is correct for real three-dimensional glasses remains an open question.

## Relation to Active Inference
The link is an analogy. Hopfield networks and related associative memories are spin-glass-like models whose stored patterns are minima in a landscape, and discussions of the free energy principle use landscapes of attractors and metastability. Glassy dynamics adds the possibility of extremely slow relaxation, history dependence and aging in such landscapes. The free energy principle does not predict glassy behaviour in the brain, and any such claim is a hypothesis about particular models.

## Related concepts
- [[broken_ergodicity]] — the failure of ensemble averaging in trapped systems
- [[free_energy_landscape]] — the landscape whose structure controls relaxation
- [[attractor_dynamics]] — stable states in recurrent systems
- [[non_equilibrium_thermodynamics]] — aging systems are out of equilibrium
- [[phase_transitions]] — the transition that glasses resemble but are distinct from
- [[energy_based_models]] — models defined by an energy function over states

## References
- Binder, K., & Young, A. P. (1986). Spin glasses: Experimental facts, theoretical concepts, and open questions. Reviews of Modern Physics, 58, 801-976.
- Cavagna, A. (2009). Supercooled liquids for pedestrians. Physics Reports, 476, 51-124.
- Berthier, L., & Biroli, G. (2011). Theoretical perspective on the glass transition and amorphous materials. Reviews of Modern Physics, 83, 587-645.
