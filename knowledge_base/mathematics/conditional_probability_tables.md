---
title: Conditional Probability Tables
description: Tables specifying the distribution of a discrete variable for each configuration of its parents in a Bayesian network, their size, estimation from data, and correspondence to likelihood and transition matrices.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - bayesian_networks
  - discrete_probability
  - graphical_models
semantic_relations:
  - type: relates
    links:
      - "[[bayesian_networks]]"
      - "[[conditional_probability]]"
      - "[[A_matrix]]"
      - "[[B_matrix]]"
      - "[[probabilistic_graphical_models]]"
---

# Conditional Probability Tables

## Overview
A conditional probability table (CPT) specifies the local probability model of a discrete variable in a Bayesian network. For a node $X$ with parents $\mathrm{Pa}(X)$, the table lists $P(X=x\mid \mathrm{Pa}(X)=u)$ for every value $x$ and every joint configuration $u$ of the parents. Each column, one per parent configuration, is a probability distribution over $X$ and sums to one. A node with no parents has a single column, its marginal distribution.

## Role in the model
A Bayesian network over variables $X_1,\dots,X_n$ on a directed acyclic graph defines the joint distribution as a product of local factors,
$$P(X_1,\dots,X_n)=\prod_{i=1}^{n}P\big(X_i\mid\mathrm{Pa}(X_i)\big),$$
and the CPTs are the numerical content of those factors. Inference, such as computing a posterior over some variables given evidence, uses these tables through methods like variable elimination or belief propagation.

## Size and estimation
A variable with $r$ values and parents with $r_1,\dots,r_k$ values has $r\prod_j r_j$ entries, of which $(r-1)\prod_j r_j$ are free parameters, since columns sum to one. The table grows exponentially with the number of parents. Compact alternatives, such as noisy-OR models, decision trees or parametric forms, reduce the number of parameters. When data are fully observed, maximum likelihood estimates are normalised counts, and a Dirichlet prior on each column, which is conjugate to the categorical distribution, gives a posterior with pseudo-counts added to the observed counts. With missing data, expectation-maximisation is the usual approach.

## Relation to Active Inference
The correspondence is direct. In discrete-state Active Inference models, the likelihood matrix $A$ is a CPT for $P(o\mid s)$, the transition matrices $B$ are CPTs for $P(s_{t+1}\mid s_t,a)$ with one table per action, and the initial-state vector $D$ is a prior table. Learning in these models places Dirichlet priors on the columns of $A$ and $B$ and accumulates concentration parameters by counting, exactly as for CPTs. These identities follow from how the models are specified. The preference vector $C$ is a distribution over outcomes and is not a conditional table.

## Related concepts
- [[bayesian_networks]] — the models whose local factors are CPTs
- [[conditional_probability]] — the quantity tabulated
- [[A_matrix]] — the observation likelihood as a CPT
- [[B_matrix]] — the transition model as a set of CPTs
- [[D_matrix]] — the prior table over initial states
- [[directed_acyclic_graphs]] — the graph structure that defines the parents

## References
- Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models: Principles and Techniques. MIT Press.
- Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems. Morgan Kaufmann.
- Smith, R., Friston, K. J., & Whyte, C. J. (2022). A step-by-step tutorial on active inference and its application to empirical data. Journal of Mathematical Psychology, 107, 102632.
