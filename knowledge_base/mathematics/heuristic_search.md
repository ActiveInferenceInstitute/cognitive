---
title: Heuristic Search
description: Heuristic search uses an estimate of remaining cost to guide exploration of a state space, with A* finding optimal solutions when the heuristic is admissible and consistent.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - search
  - artificial_intelligence
  - problem_solving
semantic_relations:
  - type: relates
    links:
      - "[[breadth_first]]"
      - "[[depth_first]]"
      - "[[search_strategies]]"
      - "[[shortest_path]]"
      - "[[heuristics]]"
---

# Heuristic Search

## Overview
Heuristic search, also called informed search, explores a state space using a heuristic function $h(n)$ that estimates the cost of reaching a goal from node $n$. Uninformed methods such as [[breadth_first]] and [[depth_first]] search expand nodes without reference to the goal. A heuristic lets the procedure concentrate on promising nodes and typically expands far fewer. The term is distinct from the cognitive heuristics of judgement research, which are human rules of thumb.

## A* search
A* orders the frontier by $f(n)=g(n)+h(n)$, where $g(n)$ is the cost of the best known path from the start to $n$. Two properties of $h$ matter. A heuristic is admissible if $h(n)\le h^{\ast}(n)$ for every node, where $h^{\ast}$ is the true cost to the nearest goal. It is consistent if $h(n)\le c(n,n')+h(n')$ for every successor $n'$ reached by a step of cost $c(n,n')$. Consistency implies admissibility. Hart, Nilsson and Raphael showed that with an admissible heuristic A* returns an optimal solution when step costs are positive, and with a consistent heuristic the first time a node is expanded its cost is already optimal. Setting $h=0$ gives Dijkstra's algorithm, and a more accurate admissible $h$ gives fewer expansions. In the worst case time and memory are still exponential in the solution depth, and memory is often the limit.

## Variants
- **Greedy best-first search** orders by $h(n)$ alone. It is fast but neither optimal nor complete in general.
- **Iterative-deepening A\*** performs depth-first searches with increasing bounds on $f$, which reduces memory to linear in the depth, as shown by Korf.
- **Weighted A\*** uses $f=g+wh$ with $w>1$, accepting a bounded loss of optimality for speed.

Heuristics are often derived from a relaxed version of the problem, such as straight-line distance in route finding.

## Relation to Active Inference
Planning in Active Inference scores policies by expected free energy, and tractable implementations prune or guide a tree search over action sequences by the value of branches. In this loose sense expected free energy plays a role similar to a heuristic evaluation function, and sophisticated inference searches recursively over future beliefs. The analogy does not extend to the guarantees of A*, because expected free energy is not constructed as an admissible bound on a cost-to-go.

## Related concepts
- [[breadth_first]] — uninformed search that A* generalises when $h=0$ and costs are uniform
- [[depth_first]] — memory-light uninformed search
- [[search_strategies]] — the wider class of search methods
- [[shortest_path]] — the optimisation problem A* solves
- [[heuristics]] — cognitive rules of thumb
- [[reasoning_problem_solving]] — human problem solving as search

## References
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. IEEE Transactions on Systems Science and Cybernetics, 4, 100-107.
- Korf, R. E. (1985). Depth-first iterative-deepening: An optimal admissible tree search. Artificial Intelligence, 27, 97-109.
- Russell, S., & Norvig, P. (2021). Artificial Intelligence: A Modern Approach (4th ed.). Pearson.
