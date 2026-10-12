---
title: Faithfulness
description: Faithfulness in causal graphical models is the assumption that every conditional independence in a distribution is implied by d-separation in the generating graph, making structure learning from data possible.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - causal_inference
  - graphical_models
  - conditional_independence
semantic_relations:
  - type: relates
    links:
      - "[[d_separation]]"
      - "[[directed_acyclic_graphs]]"
      - "[[markov_properties]]"
      - "[[pc_algorithm]]"
      - "[[causal_inference]]"
---

# Faithfulness

## Overview
In graphical models, faithfulness is the condition that links the independencies of a probability distribution to the structure of a graph. A distribution $P$ is faithful to a directed acyclic graph $G$ if every conditional independence that holds in $P$ is entailed by $G$ through [[d_separation]]. The word also has an unrelated technical sense in category theory, where a faithful functor is injective on morphisms; that sense is not meant here.

## Definition and relation to the Markov condition
The causal Markov condition states that if $X$ and $Y$ are d-separated by a set $Z$ in $G$, then $X\perp\!\!\!\perp Y\mid Z$ in $P$. Faithfulness is the converse:
$$X\perp\!\!\!\perp Y\mid Z \text{ in } P \;\Longrightarrow\; X \text{ and } Y \text{ are d-separated by } Z \text{ in } G.$$
Together the two conditions say that the independencies of $P$ are exactly those read from the graph.

Unfaithful distributions arise when causal influences cancel. For a linear model with $X\to Y\to Z$ and $X\to Z$, with path coefficients $a$, $b$ and $c$, the total effect of $X$ on $Z$ is $c+ab$. If $c=-ab$ then $X$ and $Z$ are marginally independent although the graph contains a directed path between them. For linear Gaussian models and for discrete models with a fixed graph, the parameter values that produce such unfaithful distributions form a set of Lebesgue measure zero, which is the usual justification for assuming faithfulness. Near-cancellations can still produce dependencies too weak to detect in finite samples.

## Use in structure learning
Constraint-based algorithms such as the PC algorithm recover a causal graph from conditional independence tests, relying on faithfulness for correctness. Without it, an observed independence cannot be interpreted as absence of an edge. Even with it, only the Markov equivalence class can be identified from observational data, and the guarantees assume sufficiently large samples and reliable tests.

## Relation to Active Inference
Markov blankets in Active Inference are defined by conditional independence, and approaches to learning model structure or to identifying blankets from data rely on the same independence-to-graph reasoning. In that sense faithfulness is a background assumption of those methods, not a result of the theory, and it can fail for deterministic or finely tuned systems.

## Related concepts
- [[d_separation]] — the graphical criterion for independence
- [[directed_acyclic_graphs]] — the graph class to which faithfulness applies
- [[markov_properties]] — the forward direction linking graphs to independence
- [[pc_algorithm]] — a learning algorithm that assumes it
- [[conditional_independence_tests]] — the statistical tools used to check independence
- [[markov_blanket]] — a structure defined by conditional independence

## References
- Spirtes, P., Glymour, C., & Scheines, R. (2000). Causation, Prediction, and Search (2nd ed.). MIT Press.
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference (2nd ed.). Cambridge University Press.
