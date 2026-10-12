---
title: Shortest Path
description: The problem of finding a minimum-weight route between vertices of a graph, the main algorithms, and its appearance in ant trail formation and in planning as a deterministic limit.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - graph_algorithms
  - optimisation
  - ant_colonies
semantic_relations:
  - type: relates
    links:
      - "[[graph_theory]]"
      - "[[dynamic_programming]]"
      - "[[traveling_salesman]]"
      - "[[ant_colony_optimization]]"
      - "[[trail_pheromones]]"
---

# Shortest Path

## Overview
Given a weighted graph $G=(V,E,w)$ with edge weights $w:E\to\mathbb{R}$, a shortest path from a source $s$ to a target $t$ is a path whose total weight $\sum_{e\in P}w(e)$ is minimal over all paths from $s$ to $t$. The problem is a basic building block of routing, planning and network analysis, and its structure is captured by [[graph_theory]].

## Algorithms
The central property is optimal substructure: if $d(v)$ denotes the shortest distance from $s$ to $v$, then
$$d(v)=\min_{u:(u,v)\in E}\bigl[d(u)+w(u,v)\bigr],\qquad d(s)=0,$$
which is a Bellman equation, so the problem is a special case of [[dynamic_programming]].

- **Dijkstra's algorithm** handles non-negative weights. With a binary heap it runs in $O((|V|+|E|)\log|V|)$ time, and with a Fibonacci heap in $O(|E|+|V|\log|V|)$.
- **Bellman-Ford** allows negative weights, detects negative cycles reachable from the source, and runs in $O(|V||E|)$.
- **Floyd-Warshall** computes all-pairs distances in $O(|V|^3)$.
- **A\*** adds an admissible heuristic to focus the search toward the target.

The related [[traveling_salesman]] problem, which requires visiting every vertex, is NP-hard, whereas single-pair shortest path is solvable in polynomial time.

## Ant trails
In double-bridge experiments with Argentine ants (Goss et al., 1989; Deneubourg et al., 1990), colonies given two routes to food settle on one branch, and when the routes differ in length they come to use the shorter one. The mechanism needs no individual comparison of routes: ants lay pheromone, ants on the shorter branch complete round trips sooner, and the resulting faster reinforcement of that branch is amplified by positive feedback (see [[trail_pheromones]] and [[collective_behavior_ants]]). The outcome is a stochastic, self-organised approximation to a shortest path, not a guaranteed optimum, and it inspired [[ant_colony_optimization]].

## Relation to Active Inference
In planning, a deterministic environment with a single goal state and a fixed cost per step reduces to a shortest-path problem, and dynamic programming then yields the optimal policy. Active Inference generalises this to stochastic transitions and to scoring policies by expected free energy, which adds an information-seeking term that pure shortest-path cost lacks. The connection is therefore a limiting case, not an equivalence.

## Related concepts
- [[graph_theory]] — the mathematical setting
- [[dynamic_programming]] — the optimality principle behind the algorithms
- [[traveling_salesman]] — the harder tour-finding relative
- [[ant_colony_optimization]] — pheromone-based heuristic search
- [[trail_following]] — the behaviour that implements path choice in ants

## References
- Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. Numerische Mathematik, 1, 269-271.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press.
- Goss, S., Aron, S., Deneubourg, J. L., & Pasteels, J. M. (1989). Self-organized shortcuts in the Argentine ant. Naturwissenschaften, 76, 579-581.
- Deneubourg, J. L., Aron, S., Goss, S., & Pasteels, J. M. (1990). The self-organizing exploratory pattern of the Argentine ant. Journal of Insect Behavior, 3, 159-168.
