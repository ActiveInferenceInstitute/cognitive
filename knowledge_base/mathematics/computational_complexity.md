---
title: Computational Complexity
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - complexity_theory
  - np_hardness
  - inference
semantic_relations:
  - type: relates
    links:
      - "[[complexity_measures]]"
      - "[[inference_algorithms]]"
      - "[[model_complexity]]"
      - "[[policy_search_methods]]"
wikidata: Q5157286
---

# Computational Complexity

## Overview
Computational complexity theory classifies problems by the resources (time, memory) any algorithm needs to solve them as input size grows. It separates what is solvable in principle from what is feasible, and it frames why exact Bayesian inference and optimal planning are generally intractable.

## Definition / Formalism
A decision problem is in P if a deterministic algorithm decides it in time polynomial in the input size, and in NP if a proposed solution can be verified in polynomial time. A problem is NP-hard if every NP problem reduces to it in polynomial time; NP-complete if it is also in NP (Cook, 1971, for SAT). PSPACE contains problems solvable with polynomial memory, and #P counts the solutions of NP problems. The inclusions $\mathrm P\subseteq\mathrm{NP}\subseteq\mathrm{PSPACE}$ are known; whether $\mathrm P=\mathrm{NP}$ is open.

Known results relevant to probabilistic modelling:
- Exact inference in general Bayesian networks is NP-hard (Cooper, 1990) and computing marginal probabilities is #P-complete (Roth, 1996). On trees, or graphs of low treewidth, belief propagation and junction-tree methods are polynomial in the number of nodes and exponential only in treewidth.
- Finite-horizon POMDP policy existence is PSPACE-complete (Papadimitriou and Tsitsiklis, 1987).
- Exhaustive policy evaluation scales as $|\mathcal A|^T$: three actions over a horizon of five steps give $3^5=243$ policies.

## Role in Active Inference
These results explain design choices rather than being consequences of the theory. Free energy minimisation turns inference into optimisation, but it does not make a hard problem easy: the approximation quality (mean-field, Bethe) is what is bought, and the optimisation of $F$ is typically non-convex. Policy spaces are kept manageable by restricting the policy set, by planning over a short horizon, by tree search with pruning in sophisticated inference, and by factorised beliefs, whose message updates cost roughly a matrix-vector product per factor. Claims that biological or artificial agents "solve" intractable inference exactly are not supported; the principled claim is approximate inference under resource limits. The vault page on model complexity concerns the KL complexity term in free energy, which is a different notion from the algorithmic notion here.

## Related concepts
- [[complexity_measures]] — information-theoretic and structural measures of complexity
- [[model_complexity]] — the complexity term in free energy and model selection
- [[inference_algorithms]] — approximate methods that trade exactness for tractability
- [[belief_propagation]] — exact on trees, approximate on loopy graphs
- [[policy_search_methods]] — heuristics for exponentially large policy spaces
- [[proof_theory]] — decidability and proof-search limits

## References
- Cook (1971), The complexity of theorem-proving procedures.
- Cooper (1990), The computational complexity of probabilistic inference using Bayesian belief networks, Artificial Intelligence 42.
- Papadimitriou & Tsitsiklis (1987), The complexity of Markov decision processes, Mathematics of Operations Research.
- Roth (1996), On the hardness of approximate reasoning, Artificial Intelligence 82.
