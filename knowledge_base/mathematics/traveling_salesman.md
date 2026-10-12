---
title: Traveling Salesman
description: The combinatorial optimisation problem of finding the shortest closed tour through a set of cities, its NP-hardness, exact and approximation algorithms, and its role as a benchmark for swarm methods.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - combinatorial_optimisation
  - computational_complexity
  - graph_algorithms
semantic_relations:
  - type: relates
    links:
      - "[[shortest_path]]"
      - "[[ant_colony_optimization]]"
      - "[[computational_complexity]]"
      - "[[graph_theory]]"
      - "[[swarm_intelligence]]"
      - "[[optimization_algorithms]]"
---

# Traveling Salesman

## Overview
The traveling salesman problem (TSP) asks for the cheapest closed route that visits each of $n$ cities exactly once and returns to the start. In graph terms, given a complete graph with edge costs $c_{ij}$, find a Hamiltonian cycle of minimum total cost. In the symmetric case there are $(n-1)!/2$ distinct tours, so enumeration is infeasible beyond very small $n$.

## Complexity and algorithms
The decision version of the TSP is NP-complete, which follows from the NP-completeness of the Hamiltonian cycle problem shown by Karp (1972), and the optimisation version is NP-hard (Garey and Johnson, 1979). It differs from the [[shortest_path]] problem, which asks for a route between two given vertices and is solvable in polynomial time.

- **Exact methods.** The Held-Karp dynamic programme runs in $O(n^2 2^n)$ time and $O(n2^n)$ space, far better than $n!$. Practical solvers use integer programming, with subtour-elimination constraints, cutting planes and branch-and-bound; they have solved instances with tens of thousands of cities, and the state of the art is described by Applegate and colleagues.
- **Approximation.** For metric instances that satisfy the triangle inequality, Christofides' algorithm guarantees a tour at most $3/2$ times the optimum. Without the triangle inequality, no constant-factor approximation exists unless P = NP. For Euclidean instances, Arora gave a polynomial-time approximation scheme.
- **Heuristics.** Nearest-neighbour construction, 2-opt and Lin-Kernighan local search find good tours quickly without optimality guarantees.

## Swarm methods
The TSP became the first benchmark for ant colony optimisation. In the Ant System of Dorigo, Maniezzo and Colorni, artificial ants build tours with probabilities weighted by pheromone strength and by inverse distance, and shorter tours deposit more pheromone, mimicking the positive feedback in real ant trails (see [[ant_colony_optimization]] and [[swarm_intelligence]]). These methods are competitive heuristics on some instance classes but are not provably optimal, and specialised local-search solvers often outperform them on large instances.

## Relation to Active Inference
The TSP is a planning problem that Active Inference could in principle address by treating the sequence of cities as a policy, but the number of policies grows factorially, so the expected-free-energy evaluation would face the same combinatorial barrier. The link is as a benchmark of sequential planning under constraints, and not a direct application of the theory. In biology, foragers that visit multiple food sites (traplining) have been studied as approximate solvers of tour problems.

## Related concepts
- [[shortest_path]] — the polynomial-time relative that asks for a single route
- [[ant_colony_optimization]] — pheromone-based heuristic evaluated on this problem
- [[computational_complexity]] — the framework in which its hardness is stated
- [[graph_theory]] — Hamiltonian cycles and weighted graphs
- [[swarm_intelligence]] — the family of collective methods

## References
- Applegate, D. L., Bixby, R. E., Chvátal, V., & Cook, W. J. (2006). The Traveling Salesman Problem: A Computational Study. Princeton University Press.
- Held, M., & Karp, R. M. (1962). A dynamic programming approach to sequencing problems. Journal of the Society for Industrial and Applied Mathematics, 10, 196-210.
- Garey, M. R., & Johnson, D. S. (1979). Computers and Intractability: A Guide to the Theory of NP-Completeness. W. H. Freeman.
- Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: Optimization by a colony of cooperating agents. IEEE Transactions on Systems, Man, and Cybernetics, Part B, 26, 29-41.
- Karp, R. M. (1972). Reducibility among combinatorial problems. In R. E. Miller & J. W. Thatcher (Eds.), Complexity of Computer Computations (pp. 85-103). Plenum Press.
