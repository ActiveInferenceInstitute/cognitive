---
title: Constraint Propagation
description: Constraint propagation prunes the possible values of variables by enforcing local consistency among constraints, as in arc consistency, and in a broader sense describes how constraints at one level shape another.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - constraint_satisfaction
  - algorithms
  - inference
semantic_relations:
  - type: relates
    links:
      - "[[constraint_satisfaction]]"
      - "[[belief_propagation]]"
      - "[[message_passing]]"
      - "[[depth_first]]"
      - "[[emergence_self_organization]]"
---

# Constraint Propagation

## Overview
Constraint propagation is a family of inference procedures for constraint satisfaction problems. A problem consists of variables, a finite domain of values for each, and constraints restricting which combinations of values are allowed. Propagation uses each constraint to remove values that cannot appear in any solution, and repeats until no further values can be removed. The result is a smaller problem equivalent to the original.

## Local consistency
The most common form is arc consistency. A variable $X$ is arc consistent with a constraint on $X$ and $Y$ if every value in the domain of $X$ has a supporting value in the domain of $Y$ that satisfies the constraint. The AC-3 algorithm of Mackworth keeps a queue of arcs, revises the domain of $X$ by deleting unsupported values, and re-queues the arcs affected by any deletion. With $e$ binary constraints and domain size $d$ its worst-case time is $O(ed^3)$, and later algorithms reach $O(ed^2)$.

Propagation is sound, because it never removes a value that belongs to a solution, but it is generally incomplete. A problem can be arc consistent and still have no solution. Propagation is therefore combined with search: a solver assigns a variable by [[depth_first]] backtracking, propagates, and backtracks if some domain becomes empty. If the constraint graph is a tree, enforcing arc consistency in a suitable order allows a solution to be found without backtracking.

## Constraints across levels
In writing on emergence and self-organisation the phrase is used more loosely. It refers to the way restrictions imposed by a collective or higher-level structure narrow the states that lower-level components can occupy, and how local restrictions in turn spread through a network. The algorithmic definition above is the precise version of this idea.

## Relation to Active Inference
The relation is close in structure. Belief propagation and variational message passing also pass local messages along a graph until beliefs are consistent. A hard constraint can be written as a factor taking values zero or one, and propagation then corresponds to a deterministic limit of message passing in which only support is tracked, not probability. This is a correspondence between algorithms and not a claim that Active Inference agents run arc-consistency routines.

## Related concepts
- [[constraint_satisfaction]] — the problem class that propagation simplifies
- [[belief_propagation]] — the probabilistic counterpart on graphs
- [[message_passing]] — the general scheme of local exchange
- [[factor_graphs]] — a representation containing constraints as factors
- [[depth_first]] — the search procedure that typically accompanies propagation
- [[emergence_self_organization]] — the cross-level sense of constraint in complex systems

## References
- Mackworth, A. K. (1977). Consistency in networks of relations. Artificial Intelligence, 8, 99-118.
- Dechter, R. (2003). Constraint Processing. Morgan Kaufmann.
- Russell, S., & Norvig, P. (2021). Artificial Intelligence: A Modern Approach (4th ed.). Pearson.
