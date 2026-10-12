---
title: Particle Swarm Optimization
description: A population-based stochastic optimisation method in which candidate solutions move through a search space guided by their own best positions and the swarm's best, with the update rule and stability analysis.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - optimization
  - swarm_intelligence
  - metaheuristics
semantic_relations:
  - type: relates
    links:
      - "[[swarm_intelligence]]"
      - "[[optimization_algorithms]]"
      - "[[evolutionary_algorithms]]"
      - "[[stochastic_optimization]]"
      - "[[random_search]]"
---

# Particle Swarm Optimization

## Overview
Particle swarm optimization (PSO) was introduced by Kennedy and Eberhart (1995) as a method for minimising a function $f:\mathbb{R}^d\to\mathbb{R}$ without using gradients. A population of particles moves through the search space, and each particle adjusts its movement using its own best-found position and the best position found by the swarm or its neighbours. The method was motivated by models of flocking and social behaviour, but it is an optimisation heuristic and not a model of any particular animal.

## Update rule
Particle $i$ has position $x_i$, velocity $v_i$, a personal best $p_i$ and a swarm (or neighbourhood) best $g$. At each iteration,
$$v_i\leftarrow\omega\,v_i+c_1r_1\odot(p_i-x_i)+c_2r_2\odot(g-x_i),\qquad x_i\leftarrow x_i+v_i,$$
where $r_1,r_2$ are vectors of independent uniform random numbers on $[0,1]$, $\odot$ is componentwise multiplication, $\omega$ is an inertia weight and $c_1,c_2$ are acceleration coefficients. The second term is the cognitive pull toward the particle's own experience and the third the social pull toward the swarm. After evaluating $f$, the bests $p_i$ and $g$ are updated.

Clerc and Kennedy (2002) analysed the dynamics of a simplified deterministic system and derived constriction coefficients that keep particle trajectories from diverging without needing an explicit velocity limit. The choice of $\omega$, $c_1$ and $c_2$ controls the balance between exploration and exploitation, so the method has the same tradeoff as other stochastic search procedures. Topology of the neighbourhood also matters: a fully connected swarm converges quickly but is more likely to stall in a local minimum, while sparser neighbourhoods slow the spread of information.

PSO requires only function evaluations and is simple to parallelise, and it is widely applied to continuous problems of moderate dimension. It offers no general guarantee of convergence to a global optimum, and its performance on a given problem is usually assessed empirically against alternatives such as evolutionary algorithms and random search.

## Relation to Active Inference
Particle swarm optimization has no derivation from Active Inference. It can be placed alongside other collective-search heuristics in discussions of swarm intelligence, where individual agents with simple rules produce useful group behaviour. Active Inference treatments of collective behaviour model agents with generative models and beliefs, which is a different formulation from fixed update rules.

## Related concepts
- [[swarm_intelligence]] — collective problem solving by simple agents
- [[optimization_algorithms]] — the wider family of numerical methods
- [[evolutionary_algorithms]] — another population-based approach
- [[stochastic_optimization]] — optimisation with randomness in the search
- [[random_search]] — baseline against which PSO is compared
- [[exploration_exploitation]] — the tradeoff governed by the coefficients

## References
- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. Proceedings of the IEEE International Conference on Neural Networks, 4, 1942-1948.
- Clerc, M., & Kennedy, J. (2002). The particle swarm - explosion, stability, and convergence in a multidimensional complex space. IEEE Transactions on Evolutionary Computation, 6, 58-73.
- Bonabeau, E., Dorigo, M., & Theraulaz, G. (1999). Swarm Intelligence: From Natural to Artificial Systems. Oxford University Press.
