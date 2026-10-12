---
title: Broken Ergodicity
description: Broken ergodicity describes systems whose phase space splits into regions that cannot be crossed on observable timescales, so time averages differ from ensemble averages, as in glasses and ferromagnets.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistical_physics
  - ergodicity
  - nonequilibrium
semantic_relations:
  - type: relates
    links:
      - "[[ergodicity]]"
      - "[[ergodic_theory]]"
      - "[[non_ergodic_systems]]"
      - "[[glassy_dynamics]]"
      - "[[free_energy_principle]]"
---

# Broken Ergodicity

## Overview
A system is ergodic when the long-time average of an observable along a single trajectory equals its average over the equilibrium ensemble. Broken ergodicity, a term made precise by R. G. Palmer, refers to systems for which this equality fails on the timescales of an experiment or observation. The accessible phase space is partitioned into components, and the system visits only the one in which it started because transitions between components are exponentially slow in system size or are blocked by barriers.

## Mechanism
Consider a ferromagnet below its critical temperature. The Hamiltonian is symmetric under flipping all spins, so the equilibrium ensemble has zero mean magnetisation. A single sample, however, remains in a state with positive or negative magnetisation for times that grow exponentially with the number of spins. Time averages then equal averages over one ergodic component, not over the whole ensemble. This is spontaneous symmetry breaking viewed dynamically.

Two situations can be distinguished. In a ferromagnet the barriers between components diverge in the thermodynamic limit. In glasses and spin glasses, by contrast, barriers are finite and a hierarchy of relaxation times appears, so the apparent number of accessible components depends on the observation time $t_{\mathrm{obs}}$. Quantities such as the susceptibility measured on short and long timescales then differ, and responses depend on the age of the sample.

The consequence is that ensemble calculations must be restricted to a single component, or the dynamics must be treated explicitly, if they are to describe what one system actually does.

## Relation to Active Inference
The free energy principle is usually stated for systems with a nonequilibrium steady-state density and is sometimes described in terms of ergodic random dynamical systems, so the status of ergodicity bears on its scope. Where a system occupies a single metastable component for the observation time, the relevant density is the one conditional on that component. The link is a conceptual caveat about the assumptions of the formalism, not a derivation, and the treatment of systems that switch between components is an open question in that literature.

## Related concepts
- [[ergodicity]] — the property whose failure defines the topic
- [[ergodic_theory]] — the mathematical framework for time and space averages
- [[non_ergodic_systems]] — systems where ergodic averaging is not valid
- [[glassy_dynamics]] — slow, history-dependent relaxation in disordered systems
- [[symmetry_breaking]] — the equilibrium counterpart of component selection
- [[phase_transitions]] — the settings in which ergodicity typically breaks

## References
- Palmer, R. G. (1982). Broken ergodicity. Advances in Physics, 31, 669-735.
- Binder, K., & Young, A. P. (1986). Spin glasses: Experimental facts, theoretical concepts, and open questions. Reviews of Modern Physics, 58, 801-976.
