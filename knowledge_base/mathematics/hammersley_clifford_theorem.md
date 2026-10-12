---
title: Hammersley-Clifford Theorem
description: The theorem stating that a strictly positive joint distribution satisfies the Markov properties of an undirected graph if and only if it factorises into functions of the graph's cliques, which identifies Markov random fields with Gibbs random fields.
type: mathematical_concept
status: stable
created: 2026-10-12
updated: 2026-10-12
tags:
  - graphical_models
  - markov_random_fields
  - statistical_physics
  - probability
aliases:
  - Hammersley-Clifford
  - Fundamental theorem of random fields
semantic_relations:
  - type: relates
    links:
      - "[[markov_random_fields]]"
      - "[[gibbs_distributions]]"
      - "[[markov_properties]]"
      - "[[conditional_independence]]"
      - "[[probabilistic_graphical_models]]"
      - "[[partition_function]]"
---

# Hammersley-Clifford Theorem

## Overview
The Hammersley-Clifford theorem links two ways of describing a joint distribution over variables attached to the nodes of an undirected graph. One is local: each variable is conditionally independent of the rest given its neighbours. The other is a factorisation: the joint is a product of positive functions, each depending only on a clique. For strictly positive distributions the two descriptions are equivalent. The result was announced by J. M. Hammersley and P. Clifford in an unpublished manuscript of 1971 and was proved in print by several authors, including Grimmett (1973) and Besag (1974).

## Statement
Let $G=(V,E)$ be a finite undirected graph and $X=(X_v)_{v\in V}$ a vector of random variables with finitely many values and joint probability mass function $p$ with $p(x)>0$ for every configuration $x$. Then $p$ satisfies the (pairwise, local, or global) Markov property with respect to $G$ if and only if it can be written as
$$p(x)=\frac{1}{Z}\prod_{C\in\mathcal{C}}\psi_C(x_C),$$
where $\mathcal{C}$ is the set of cliques (or just the maximal cliques) of $G$, each $\psi_C>0$ depends only on the coordinates in $C$, and $Z$ is the normalising constant. Writing $\psi_C=\exp(-\phi_C)$ gives the Gibbs form $p(x)\propto\exp(-\sum_C\phi_C(x_C))$ used in [[gibbs_distributions]].

## Ideas behind the proof
Factorisation implies the Markov properties directly, because conditioning on the neighbours of a node leaves only factors that involve that node's clique, and these cancel in the conditional. The converse uses positivity. Fix a reference configuration $x^*$ and expand $\log p(x)/p(x^*)$ by Möbius inversion over subsets $A\subseteq V$ into terms $V_A(x_A)$ that depend only on the coordinates in $A$. The pairwise Markov property forces $V_A=0$ whenever $A$ contains two non-adjacent nodes, so only terms supported on cliques remain.

## Role of positivity
The theorem is false without strict positivity. Distributions with zero-probability configurations can satisfy the Markov properties of a graph without factorising over its cliques. Positivity also gives the equivalence of the pairwise, local and global Markov properties, which in general are only successively weaker (see [[markov_properties]]).

## Consequences
- Models can be specified through local potentials without checking global consistency, since any positive clique potentials define a valid Markov random field.
- The parameters of such models are not unique, because potentials can be rescaled by constants and absorbed into $Z$.
- Computation of $Z$ is typically the hard step (see [[partition_function]]), which motivates Gibbs sampling and variational approximations.

## Relation to Active Inference
The link is structural. Active Inference models are mostly written as directed generative models, so the theorem applies most directly to undirected formulations, such as Markov random fields or factor-graph treatments of message passing. The Gibbs form it guarantees is the same exponential-of-an-energy structure that appears when free-energy treatments write a steady-state density as $p\propto e^{-\text{surprise}}$. Whether a given physical system's blanket structure corresponds to a graph with this factorisation is an empirical and modelling question, not a consequence of the theorem.

## References
- Besag, J. (1974). Spatial interaction and the statistical analysis of lattice systems. Journal of the Royal Statistical Society: Series B, 36(2), 192-225. doi:10.1111/j.2517-6161.1974.tb00999.x
- Grimmett, G. R. (1973). A theorem about random fields. Bulletin of the London Mathematical Society, 5(1), 81-84. doi:10.1112/blms/5.1.81
- Lauritzen, S. L. (1996). Graphical Models. Oxford University Press (Clarendon Press).
- Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models: Principles and Techniques. MIT Press.
