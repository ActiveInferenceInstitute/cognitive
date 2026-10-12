---
title: Bayesian Mechanics
description: A probabilistic mechanics for systems partitioned by a Markov blanket, in which the internal states of a system at steady state parametrise beliefs about its external states.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
aliases:
  - Bayesian mechanics
tags:
  - free_energy_principle
  - statistical_physics
  - markov_blanket
  - stochastic_processes
semantic_relations:
  - type: foundation
    links:
      - "[[markov_blankets]]"
      - "[[non_equilibrium_steady_state]]"
      - "[[stochastic_processes]]"
  - type: relates
    links:
      - "[[free_energy_principle]]"
      - "[[variational_free_energy]]"
      - "[[fokker_planck]]"
      - "[[information_geometry]]"
      - "[[kl_divergence]]"
---

# Bayesian Mechanics

## Overview
Bayesian mechanics is the name given in the free energy principle literature to a mathematical toolkit for describing the dynamics of a system as if its internal states tracked beliefs about the states of its surroundings. It is a probabilistic mechanics: it works with probability densities over states and their paths, in the way statistical mechanics does, and adds a partition of the system into parts that makes an inferential reading possible. Ramstead et al. (2023) define it as a set of tools for modelling systems divided into "particles", where the internal states of a particle encode the parameters of beliefs about quantities that characterise its environment.

## Construction
The starting point is a random dynamical system with a non-equilibrium steady-state density $p^*(x)$, whose state is partitioned into four sets:

- external states $\eta$ (the environment),
- sensory states $s$ and active states $a$, together the blanket states $b=(s,a)$,
- internal states $\mu$.

The partition is a Markov blanket: external and internal states are conditionally independent given the blanket, $\eta \perp \mu \mid b$. Under this condition (see [[markov_blankets]]), the most likely internal state given the blanket, $\bar\mu(b)$, determines through a map a probability density over external states, $q_{\bar\mu(b)}(\eta)$, which can be read as a belief held by the internal states. A variational free energy

$$F(b,\mu)=D_{\mathrm{KL}}\big[q_\mu(\eta)\,\|\,p^*(\eta\mid b)\big]-\ln p^*(b)$$

is then a function of blanket and internal states, and the expected flow of the internal states at steady state can be written as a gradient flow on $F$. Da Costa et al. (2021) obtain this for stationary processes, and further split the blanket into sensory and active states, after which the active states can be read as performing active inference and well-known forms of stochastic control such as PID control.

## Variants
Ramstead et al. (2023) review three ways in which the free energy principle has been applied to particular systems, which they call path-tracking, mode-tracking and mode-matching. They also examine a duality between the free energy principle and the constrained maximum entropy principle.

## Relation to Active Inference
Bayesian mechanics is the physical layer beneath [[free_energy_principle]]: it supplies the reading under which a variational free energy $F$ appears as a quantity that a sparsely coupled steady-state system can be said to minimise. The word "belief" is a statement about parametrisation, namely that internal states fix a density over external states, and not a claim that the system has a mind. The programme is contested. Biehl, Pollock and Kanai (2021) question some of its technical steps, and Aguilera et al. (2022) examine how restrictive its assumptions on coupling and noise are; Friston (2022) replied to the second paper. Whether real organisms satisfy the required conditions is an empirical question that the formalism does not settle.

## Related concepts
- [[markov_blankets]] — the partition on which the construction rests
- [[non_equilibrium_steady_state]] — the density $p^*$ around which the dynamics are described
- [[variational_free_energy]] — the quantity $F$ above
- [[information_geometry]] — the geometry of the belief densities $q_\mu$
- [[kl_divergence]] — the divergence inside $F$

## References
- Ramstead, M. J. D., Sakthivadivel, D. A. R., Heins, C., Koudahl, M., Millidge, B., Da Costa, L., Klein, B., & Friston, K. J. (2023). On Bayesian mechanics: a physics of and by beliefs. Interface Focus, 13(3). https://doi.org/10.1098/rsfs.2022.0029
- Da Costa, L., Friston, K., Heins, C., & Pavliotis, G. A. (2021). Bayesian mechanics for stationary processes. Proceedings of the Royal Society A, 477(2256). https://doi.org/10.1098/rspa.2021.0518
- Friston, K., Da Costa, L., Sajid, N., Heins, C., Ueltzhoffer, K., Pavliotis, G. A., & Parr, T. (2023). The free energy principle made simpler but not too simple. Physics Reports, 1024, 1-29. https://doi.org/10.1016/j.physrep.2023.07.001
- Aguilera, M., Millidge, B., Tschantz, A., & Buckley, C. L. (2022). How particular is the physics of the free energy principle? Physics of Life Reviews, 40, 24-50. https://doi.org/10.1016/j.plrev.2021.11.001
- Biehl, M., Pollock, F. A., & Kanai, R. (2021). A technical critique of some parts of the free energy principle. Entropy, 23(3), 293. https://doi.org/10.3390/e23030293
- Friston, K. (2022). Very particular: Comment on "How particular is the physics of the free energy principle?" Physics of Life Reviews, 41, 58-60. https://doi.org/10.1016/j.plrev.2022.05.002
