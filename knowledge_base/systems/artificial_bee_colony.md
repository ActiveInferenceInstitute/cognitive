---
title: Artificial Bee Colony
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - swarm_intelligence
  - optimization
  - metaheuristics
  - bees
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/biology/bee_colony_organization]]"
      - "[[knowledge_base/biology/bee_communication_systems]]"
      - "[[knowledge_base/biology/foraging_optimization]]"
      - "[[knowledge_base/systems/ant_colony_optimization]]"
---

# Artificial Bee Colony

## Overview
The artificial bee colony (ABC) algorithm is a population-based metaheuristic for continuous numerical optimization that abstracts the foraging behavior of honeybees. It was proposed by Dervis Karaboga in 2005 and developed with Basturk in 2007.

## Mechanism
ABC maintains a set of food sources, each a candidate solution vector, with nectar amount representing fitness. The colony is divided into three roles. Employed bees are each bound to one food source: they generate a neighbor candidate by perturbing one dimension, v_ij = x_ij + phi_ij * (x_ij - x_kj), where k is a randomly chosen other source and phi_ij is uniform in [-1, 1], and keep the better of the two (greedy selection). Onlooker bees wait at the hive, choose a source with probability proportional to its fitness, p_i = fit_i / sum(fit), and perform the same local search, so good sources get more search effort. Scout bees replace any source that has not improved after a limit number of trials with a random new solution, which supplies exploration.

The algorithm has few control parameters, essentially the colony size and the limit, which is one reason for its popularity. Its performance is competitive on some benchmark functions and weaker on others, and the neighbor equation is known to favor exploration over exploitation, which prompted variants that bias the search toward the best-so-far solution. Applications span function optimization, neural network training, and scheduling.

The correspondence to honeybee biology is loose. Real foragers recruit via the waggle dance, which encodes direction and distance (von Frisch), and decide collectively using dance duration and abandonment thresholds; ABC's fitness-proportional selection of onlookers is a simplification of that process, and the algorithm does not model dances, energetics, or nest-site decisions. Sorensen (2015) argues more generally that many nature-inspired metaheuristics differ little from existing methods beneath the metaphor, so ABC is best evaluated as an algorithm, not as a model of bees.

## Role in Active Inference
There is no formal link between ABC and Active Inference. The structural parallel is the split between exploiting known sources (employed and onlooker bees) and exploring new ones (scouts), which corresponds to the pragmatic and epistemic terms of expected free energy. An active-inference forager would obtain that balance from a single objective rather than from separate roles with a fixed trial limit. This repository treats ABC as a baseline for comparison, not as an implemented agent.

## Related concepts
- [[knowledge_base/biology/bee_colony_organization]] — colony structure and roles in real honeybees
- [[knowledge_base/biology/bee_communication_systems]] — waggle dance and recruitment communication
- [[knowledge_base/biology/foraging_optimization]] — foraging strategies of social insects
- [[knowledge_base/systems/ant_colony_optimization]] — the ant-based counterpart for discrete problems
- [[knowledge_base/cognitive/swarm_intelligence]] — the general framework
- [[knowledge_base/mathematics/exploration_exploitation]] — the trade-off that scout and onlooker roles implement

## References
- Karaboga, D., and Basturk, B. (2007). A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm. Journal of Global Optimization.
- Sorensen, K. (2015). Metaheuristics: the metaphor exposed. International Transactions in Operational Research.
