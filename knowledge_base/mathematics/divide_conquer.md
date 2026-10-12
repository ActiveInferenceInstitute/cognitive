---
title: Divide and Conquer
description: Divide and conquer solves a problem by splitting it into smaller subproblems, solving them recursively and combining the results, with running time analysed through recurrences and the master theorem.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - algorithms
  - recursion
  - problem_solving
semantic_relations:
  - type: relates
    links:
      - "[[dynamic_programming]]"
      - "[[time_complexity]]"
      - "[[computational_complexity]]"
      - "[[reasoning_problem_solving]]"
      - "[[hierarchical_models]]"
---

# Divide and Conquer

## Overview
Divide and conquer is an algorithm design strategy with three steps. The problem is divided into several smaller instances of the same problem, each instance is solved recursively until the instances are small enough to be solved directly, and the partial solutions are combined into a solution of the original. The same idea appears in everyday problem solving as decomposing a task into subtasks.

## Analysis by recurrence
If a problem of size $n$ is split into $a$ subproblems of size $n/b$, and dividing and combining cost $f(n)$, the running time satisfies
$$T(n)=a\,T(n/b)+f(n).$$
The master theorem gives the solution for common cases. Let $c=\log_b a$. If $f(n)=O(n^{c-\varepsilon})$ for some $\varepsilon>0$, then $T(n)=\Theta(n^{c})$. If $f(n)=\Theta(n^{c})$, then $T(n)=\Theta(n^{c}\log n)$. If $f(n)=\Omega(n^{c+\varepsilon})$ and a regularity condition $a f(n/b)\le k f(n)$ holds for some $k<1$, then $T(n)=\Theta(f(n))$. Some recurrences, such as those with unequal splits, fall outside the theorem and are solved by other methods.

## Examples
- **Merge sort** has $a=b=2$ and $f(n)=\Theta(n)$ for merging, which gives $\Theta(n\log n)$.
- **Binary search** has $a=1$, $b=2$ and constant $f$, which gives $\Theta(\log n)$.
- **Strassen's matrix multiplication** replaces eight recursive products by seven, giving $\Theta(n^{\log_2 7})\approx\Theta(n^{2.81})$.
- **The fast Fourier transform** splits a transform of length $n$ into two of length $n/2$ and runs in $\Theta(n\log n)$.

Correctness is normally proved by induction on problem size. Parallelism is natural because subproblems are independent. The strategy is ineffective when the subproblems overlap heavily, since the same work is repeated, which is the situation in which [[dynamic_programming]] stores and reuses results.

## Relation to Active Inference
Hierarchical generative models in Active Inference decompose perception and planning across levels and timescales, with higher levels setting context for lower ones. This resembles divide and conquer as a way to make inference tractable, and in this sense the relation is an analogy: hierarchical inference exchanges messages between levels and is not a recursive splitting of an input followed by a merge step.

## Related concepts
- [[dynamic_programming]] — the alternative when subproblems overlap
- [[time_complexity]] — the measure used in recurrence analysis
- [[computational_complexity]] — the broader theory of resource requirements
- [[reasoning_problem_solving]] — human decomposition of problems into parts
- [[hierarchical_models]] — models organised across levels of abstraction
- [[abstraction_levels]] — the levels at which a decomposition is described

## References
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press.
- Kleinberg, J., & Tardos, E. (2006). Algorithm Design. Pearson.
