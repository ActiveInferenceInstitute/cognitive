---
title: Time Complexity
description: How the running time of an algorithm grows with input size, expressed in asymptotic notation, with complexity classes, standard examples and implications for exact inference and planning.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - computational_complexity
  - algorithms
  - asymptotics
semantic_relations:
  - type: relates
    links:
      - "[[computational_complexity]]"
      - "[[exact_inference]]"
      - "[[dynamic_programming]]"
      - "[[sample_complexity]]"
      - "[[complexity_measures]]"
---

# Time Complexity

## Overview
The time complexity of an algorithm is the number of elementary steps it takes as a function $T(n)$ of the size $n$ of its input, usually in the worst case. Exact step counts depend on the machine, so analysis uses asymptotic notation. A function $f$ is $O(g)$ if there are constants $c>0$ and $n_0$ with $f(n)\le c\,g(n)$ for all $n\ge n_0$. Similarly $\Omega(g)$ gives a lower bound and $\Theta(g)$ means both bounds hold. The notation hides constant factors, which can dominate for small inputs, but it separates fundamentally different growth rates.

## Common growth rates and classes
- **Logarithmic to near-linear.** Binary search takes $O(\log n)$ steps, a scan takes $O(n)$, and comparison-based sorting takes $O(n\log n)$, which is optimal in the worst case.
- **Polynomial.** Naive multiplication of two $n\times n$ matrices takes $O(n^3)$ operations and a matrix-vector product takes $O(n^2)$. Problems solvable in polynomial time form the class P.
- **Exponential.** Enumerating all subsets or all action sequences of length $T$ with $K$ choices per step takes $O(K^T)$ time.
- **NP.** The class NP contains decision problems whose solutions can be verified in polynomial time. A problem is NP-hard if every problem in NP reduces to it in polynomial time. Whether P equals NP is open (Sipser, 2012).

## Consequences for inference and planning
Exact inference in a general discrete Bayesian network is NP-hard (Cooper, 1990). Variable elimination and message passing on a junction tree take time exponential in the treewidth of the graph, which is why exact algorithms are practical for tree-structured models and become infeasible for densely connected ones ([[exact_inference]], [[belief_propagation]]). Finite-horizon planning in partially observable Markov decision processes is PSPACE-complete in general (Papadimitriou & Tsitsiklis, 1987), so approximations are required. Dynamic programming reduces Markov sequential problems from exponential to linear in the horizon ([[dynamic_programming]]). Sample complexity counts data needed to learn, not computation ([[sample_complexity]]).

## Relation to Active Inference
Time complexity is relevant to the feasibility of Active Inference as an algorithm and, more tentatively, as a model of the brain. Evaluating expected free energy for every policy grows exponentially with planning depth, motivating tree search, pruning, and habit priors. Claims about the efficiency of neural codes ([[neural_coding]]) assume a mapping from algorithmic cost to biological resources that needs empirical support.

## Related concepts
- [[computational_complexity]] — the theory of complexity classes
- [[exact_inference]] — algorithms whose cost depends on graph structure
- [[dynamic_programming]] — reducing exponential search to polynomial
- [[sample_complexity]] — the data-based counterpart
- [[complexity_measures]] — other notions of complexity
- [[neural_coding]] — computational cost of representations

## References
- Sipser, M. (2012). Introduction to the Theory of Computation (3rd ed.). Cengage Learning.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press.
- Cooper, G. F. (1990). The computational complexity of probabilistic inference using Bayesian belief networks. Artificial Intelligence, 42, 393-405.
- Papadimitriou, C. H., & Tsitsiklis, J. N. (1987). The complexity of Markov decision processes. Mathematics of Operations Research, 12, 441-450.
