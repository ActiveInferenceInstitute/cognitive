---
title: Breadth-First Search
description: Breadth-first search explores a graph or search tree level by level using a queue, giving shortest paths in unweighted graphs at the cost of memory that grows with the frontier.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - graph_algorithms
  - search
  - problem_solving
semantic_relations:
  - type: relates
    links:
      - "[[depth_first]]"
      - "[[search_strategies]]"
      - "[[graph_theory]]"
      - "[[shortest_path]]"
      - "[[reasoning_problem_solving]]"
wikidata: Q325904
---

# Breadth-First Search

## Overview
Breadth-first search (BFS) is a strategy for traversing a graph or a search tree in which all nodes at distance $k$ from the start are visited before any node at distance $k+1$. In problem-solving terms it means considering every option at one level before going deeper into any of them. It uses a first-in, first-out queue.

## Algorithm and properties
Starting from a source vertex $s$, BFS marks $s$ as discovered and places it in a queue. It repeatedly removes the vertex at the front, examines its neighbours, and appends each undiscovered neighbour to the back of the queue, recording its parent.

- **Running time.** With an adjacency-list representation the cost is $O(|V|+|E|)$, since each vertex is enqueued once and each edge is examined a constant number of times.
- **Shortest paths.** In an unweighted graph the parent pointers define a tree in which the path from $s$ to any reachable vertex has the fewest possible edges. This holds because vertices leave the queue in non-decreasing order of distance. For weighted edges the property fails and Dijkstra's algorithm is used instead.
- **Search trees.** In a tree with branching factor $b$ whose shallowest goal lies at depth $d$, BFS is complete when $b$ is finite and optimal when all step costs are equal. It generates on the order of $b^{d}$ nodes, and it must store most of them, so memory rather than time is usually the limiting resource.

## Contrast with depth-first search
[[depth_first]] search keeps only the current path in memory but may follow a long or infinite branch and returns a path that is not necessarily shortest. BFS trades memory for guarantees on completeness and optimality. [[heuristic_search]] methods such as A* occupy the space between them.

## Relation to Active Inference
Planning in Active Inference can be viewed as search over a tree of future actions and observations, with expected free energy as the value of each branch. Exhaustive evaluation of all policies is breadth-first in character and grows exponentially with horizon, so practical schemes prune branches. The link is a computational analogy about how planning is organised, not a derivation; Active Inference does not prescribe a particular traversal order.

## Related concepts
- [[depth_first]] — the complementary traversal order that follows one path before backtracking
- [[heuristic_search]] — informed search that orders expansion by an estimated cost
- [[search_strategies]] — the wider family of search procedures
- [[shortest_path]] — the problem BFS solves for unit edge costs
- [[graph_theory]] — the structures on which BFS operates
- [[reasoning_problem_solving]] — human use of breadth and depth in solving problems

## References
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press.
- Russell, S., & Norvig, P. (2021). Artificial Intelligence: A Modern Approach (4th ed.). Pearson.
