---
title: Ant Colony Optimization
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - swarm_intelligence
  - optimization
  - metaheuristics
  - stigmergy
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/cognitive/swarm_intelligence]]"
      - "[[knowledge_base/cognitive/stigmergic_coordination]]"
      - "[[knowledge_base/biology/foraging_optimization]]"
      - "[[knowledge_base/systems/swarm_intelligence_implementation]]"
aliases:
  - ant_algorithms
---

# Ant Colony Optimization

## Overview
Ant colony optimization (ACO) is a family of metaheuristics for combinatorial optimization in which a population of artificial ants builds solutions step by step, guided by a shared memory of past successes in the form of simulated pheromone. It was introduced by Marco Dorigo in his 1992 thesis and in the Ant System of Dorigo, Maniezzo and Colorni (1996).

## Mechanism
The biological inspiration is the double-bridge experiment of Goss, Aron, Deneubourg and Pasteels (1989) on the Argentine ant: when two branches of different length connect nest and food, trail-laying and trail-following reinforcement lets most traffic converge on the shorter branch. In ACO, the problem is represented as a graph, and each artificial ant constructs a path. At node i, ant k picks next node j with probability proportional to tau_ij^alpha * eta_ij^beta over feasible choices, where tau is the pheromone level and eta is a heuristic desirability, for example the inverse of distance in the travelling salesman problem (TSP). After all ants finish, pheromone is updated by tau_ij <- (1 - rho) tau_ij + sum over ants of delta_tau_ij, with evaporation rate rho and a deposit usually proportional to the inverse of tour length for edges the ant used. Evaporation prevents premature convergence by letting the colony forget poor early choices.

Variants include Elitist Ant System, Ant Colony System (Dorigo and Gambardella 1997) with a local update and a pseudo-random proportional rule, and MAX-MIN Ant System (Stutzle and Hoos) which bounds pheromone values. Applications include the TSP, vehicle routing, quadratic assignment, scheduling, and routing in communication networks (AntNet, Di Caro and Dorigo 1998). Convergence guarantees exist for restricted variants (Gutjahr 2000; Stutzle and Dorigo 2002). Real ants differ from the algorithm in important ways, since the algorithm's ants have memory, can deposit retroactively on a completed tour, and operate in synchronous cycles.

## Role in Active Inference
The Ant_Colony Thing in this repository models foraging with active-inference agents (Nestmate) that sense pheromone, food, and nestmates, deposit pheromone, and act by free-energy minimization, in an environment where pheromone diffuses and evaporates. This differs from ACO in kind: ACO prescribes a stochastic transition rule and a fixed update, while an active-inference agent selects actions by expected free energy, which includes an epistemic term that drives it toward uncertain regions in addition to the pragmatic drive toward reward. The pheromone field plays the same structural role in both, as external shared memory, but no equivalence between the ACO rule and expected-free-energy minimization has been established here; one could be explored by checking whether the ACO exponent parameters correspond to precision.

## Related concepts
- [[knowledge_base/cognitive/swarm_intelligence]] — the broader class of decentralized collective problem-solving
- [[knowledge_base/cognitive/stigmergic_coordination]] — the indirect coordination mechanism ACO exploits
- [[knowledge_base/biology/ant_communication_systems]] — real trail and alarm signaling in ants
- [[knowledge_base/biology/chemical_ecology]] — chemistry of pheromone signals
- [[knowledge_base/systems/artificial_bee_colony]] — a companion algorithm from honeybee foraging
- [[knowledge_base/mathematics/stochastic_optimization]] — optimization with randomized search

## References
- Dorigo, M., Maniezzo, V., and Colorni, A. (1996). Ant System: optimization by a colony of cooperating agents. IEEE Transactions on Systems, Man, and Cybernetics, Part B.
- Goss, S., Aron, S., Deneubourg, J. L., and Pasteels, J. M. (1989). Self-organized shortcuts in the Argentine ant. Naturwissenschaften.
- Dorigo, M., and Stutzle, T. (2004). Ant Colony Optimization.
