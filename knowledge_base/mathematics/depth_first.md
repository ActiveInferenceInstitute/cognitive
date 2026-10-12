---
title: Depth-First Search
description: Depth-first search follows each branch of a graph or search tree as far as possible before backtracking, using a stack and linear time, and underlies topological sorting and component algorithms.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - graph_algorithms
  - search
  - backtracking
semantic_relations:
  - type: relates
    links:
      - "[[breadth_first]]"
      - "[[search_strategies]]"
      - "[[graph_theory]]"
      - "[[constraint_satisfaction]]"
      - "[[reasoning_problem_solving]]"
---

# Depth-First Search

## Overview
Depth-first search (DFS) traverses a graph or search tree by moving along one path as far as possible and only then backtracking to the most recent vertex that still has unexplored neighbours. It can be written recursively, or iteratively with an explicit last-in, first-out stack. In problem solving it corresponds to committing to one line of attack and following it to a dead end or a solution before trying alternatives.

## Algorithm and properties
DFS records, for each vertex, a discovery time and a finishing time. These timestamps classify the edges of a directed graph as tree, back, forward or cross edges.

- **Running time.** With adjacency lists DFS takes $O(|V|+|E|)$ time.
- **Memory.** In a search tree with branching factor $b$ and maximum depth $m$, DFS stores only the current path and the unexpanded siblings along it, which is $O(bm)$ nodes. This is its main advantage over [[breadth_first]] search.
- **Completeness and optimality.** DFS is not optimal, since the first goal found need not be the shallowest. In an infinite or cyclic space it can fail to terminate unless a depth limit or a visited set is used, so it is complete only for finite acyclic spaces or with cycle checking on finite graphs.
- **Applications.** A directed graph is acyclic exactly when DFS finds no back edge, and ordering vertices by decreasing finishing time gives a topological sort. DFS also yields strongly connected components and biconnected components in linear time, as shown by Tarjan.
- **Backtracking.** Constraint solvers and puzzle solvers use DFS over partial assignments, abandoning a branch as soon as a constraint is violated.

## Variants
Depth-limited search stops at a fixed depth. Iterative deepening repeats depth-limited search with increasing limits, giving the completeness and shortest-path behaviour of breadth-first search with the memory profile of DFS, at a modest repeated-work cost when the branching factor is large.

## Relation to Active Inference
Evaluating policies in Active Inference can be organised as a tree of future actions and outcomes. Depth-first expansion with early pruning of branches that have low expected value is one way to keep this evaluation tractable at long horizons. The relation is an analogy about how planning can be organised; Active Inference does not itself specify a traversal order.

## Related concepts
- [[breadth_first]] — the level-by-level alternative with different memory and optimality guarantees
- [[search_strategies]] — the broader family of uninformed and informed search
- [[constraint_satisfaction]] — problems commonly solved by backtracking DFS
- [[graph_theory]] — the setting for traversal and component algorithms
- [[reasoning_problem_solving]] — human analogues of committing to one solution path

## References
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press.
- Tarjan, R. (1972). Depth-first search and linear graph algorithms. SIAM Journal on Computing, 1, 146-160.
- Russell, S., & Norvig, P. (2021). Artificial Intelligence: A Modern Approach (4th ed.). Pearson.
