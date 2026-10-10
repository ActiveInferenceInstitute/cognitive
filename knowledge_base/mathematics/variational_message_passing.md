---
title: Variational Message Passing
description: "Mean-field variational inference expressed as local message exchange on a factor graph for conjugate-exponential models, with comparison to belief propagation and Bethe free energy."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - message_passing
  - variational_inference
  - exponential_family
semantic_relations:
  - type: relates
    links:
      - "[[message_passing]]"
      - "[[factor_graphs]]"
      - "[[variational_inference]]"
      - "[[exponential_family]]"
---

# Variational Message Passing

## Overview
Variational message passing (VMP), introduced by Winn and Bishop (2005), turns mean-field variational inference into a local algorithm on a directed graphical model. Each node updates its approximate posterior using messages from its parents and children, so inference in a new conjugate-exponential model needs no hand-derived update equations.

## Mechanism
**Mean-field update.** With $q(x)=\prod_jq_j(x_j)$, the optimal factor satisfies
$$\ln q_j^*(x_j)=\mathbb E_{q_{\setminus j}}\big[\ln p(x_j\mid\mathrm{pa}_j)+\textstyle\sum_{c\in\mathrm{ch}_j}\ln p(x_c\mid\mathrm{pa}_c)\big]+\text{const}.$$

**Conjugate-exponential models.** If every conditional is in the [[exponential_family]] and each parent-child pair is conjugate, then $\ln p(x_c\mid\mathrm{pa}_c)$ is linear in the natural statistics $u_j(x_j)$ of each parent. The update then becomes addition of natural parameters:
$$\phi_j^*=\tilde\phi_j(\{\langle u_k\rangle\}_{k\in\mathrm{pa}_j})+\sum_{c\in\mathrm{ch}_j}m_{c\to j},$$
where parent-to-child messages carry expected sufficient statistics $\langle u_j\rangle$ and child-to-parent messages are natural-parameter contributions $m_{c\to j}$ (which depend on co-parents' expected statistics).

**Scheduling.** Each update raises the evidence lower bound, so sweeping the nodes in any order converges to a local optimum of the free energy.

**Contrast with belief propagation.** Belief propagation ([[belief_propagation]]) passes marginal-like messages and fixed points correspond to stationary points of the [[bethe_free_energy]]. VMP uses expected log-factors and enforces fully factorised beliefs, so it ignores correlations between variables and typically underestimates posterior variance. Marginal message passing and constrained Bethe formulations interpolate between the two by choosing which factorisation constraints to impose; the RxInfer toolbox is built on this idea.

## Role in Active Inference
Variational message passing is the standard inference scheme in several Active Inference implementations. Discrete-state belief updating in the Friston et al. scheme is a variational message passing on the hidden Markov model factor graph, with messages from past states, future states and observations combined in log space; later work developed marginal message passing to reduce mean-field overconfidence. The scheme gives a candidate process theory: local message exchange between neuronal populations would implement perceptual inference ([[active_inference_loop]]). Whether cortical circuits implement it remains an empirical question.

## Related concepts
- [[message_passing]] — the general framework
- [[factor_graphs]] — the graph on which messages are defined
- [[belief_propagation]] — the sum-product alternative
- [[bethe_free_energy]] — what belief propagation minimises
- [[variational_free_energy]] — the objective VMP descends

## References
- Winn, J. and Bishop, C. M. (2005). Variational message passing. Journal of Machine Learning Research 6.
