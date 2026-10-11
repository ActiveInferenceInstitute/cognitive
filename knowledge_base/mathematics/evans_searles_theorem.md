---
title: Evans-Searles Theorem
description: The transient fluctuation theorem for deterministic, time-reversible systems, giving the exact ratio of probabilities of positive and negative entropy production, and its use in free energy principle arguments.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - fluctuation_theorem
  - thermodynamics
  - entropy_production
semantic_relations:
  - type: relates
    links:
      - "[[fluctuation_theorems]]"
      - "[[entropy_production]]"
      - "[[jarzynski_equality]]"
      - "[[non_equilibrium_thermodynamics]]"
      - "[[free_energy_principle]]"
---

# Evans-Searles Theorem

## Overview
The Evans-Searles theorem is a fluctuation theorem: an exact result about how likely it is, in a small system over a finite time, for entropy to be consumed rather than produced. For a deterministic, time-reversible dynamics starting from a known initial distribution, let $\Omega_t$ be the time-averaged dissipation function over a trajectory of duration $t$. The transient fluctuation theorem states
$$
\frac{P(\Omega_t=+A)}{P(\Omega_t=-A)}=e^{At}.
$$
Trajectories that dissipate positively are therefore exponentially more probable than their mirror images with negative dissipation, with the ratio growing in both the magnitude $A$ and the observation time $t$. The Second Law appears as a statement about averages, while individual short trajectories can violate it. Evans and Searles introduced the result (Evans & Searles, 1994) following earlier work on shearing steady states (Evans, Cohen & Morriss, 1993); a review is given in Evans and Searles (2002).

## Context
- **Assumptions**: time-reversibility of the microscopic dynamics, ergodic consistency of the initial distribution (nothing with positive probability maps to zero probability under reversal), and a specified initial distribution. It is valid from the start of the relaxation, not only in a long-time steady state.
- **Related results**: the Gallavotti-Cohen theorem concerns steady states, the Crooks relation concerns work in driven systems, and the Jarzynski equality can be derived from fluctuation theorems by averaging.

## Relation to Active Inference
Friston and colleagues refer to fluctuation theorems when arguing that systems which maintain a steady state with low entropy must be coupled to a dissipative environment, linking the free energy principle to non-equilibrium thermodynamics. This connection is suggestive but should be read with care. Fluctuation theorems are exact for the physical systems for which they are proved, whereas applying them to biological agents requires modelling assumptions about reversibility and the partition into agent and bath. The principle's use of them is an interpretive bridge, not a derivation of behaviour.

## Related concepts
- [[fluctuation_theorems]] — the family of results this belongs to
- [[entropy_production]] — the quantity whose fluctuations are constrained
- [[jarzynski_equality]] — a related exact non-equilibrium identity
- [[non_equilibrium_thermodynamics]] — the setting for these results
- [[thermodynamics]] — the Second Law that it refines

## References
- Evans, D. J., Cohen, E. G. D., & Morriss, G. P. (1993). Probability of second law violations in shearing steady states. Physical Review Letters, 71, 2401-2404.
- Evans, D. J., & Searles, D. J. (1994). Equilibrium microstates which generate second law violating steady states. Physical Review E, 50, 1645-1648.
- Evans, D. J., & Searles, D. J. (2002). The fluctuation theorem. Advances in Physics, 51, 1529-1585.
