---
title: Conditional Probability
description: The probability of an event given that another has occurred, with its definition, chain rule, total probability, link to Bayes' theorem, and its role in the likelihood and transition arrays of Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - probability
  - bayesian_inference
  - foundations
semantic_relations:
  - type: relates
    links:
      - "[[probability_theory]]"
      - "[[bayes_theorem]]"
      - "[[conditional_independence]]"
      - "[[random_variables]]"
wikidata: Q327069
---

# Conditional Probability

## Overview
Conditional probability measures how likely an event is once another event is known to have occurred. It is the basic operation by which probabilistic models incorporate evidence, and every generative model is a product of conditional distributions.

## Definition and rules
For events $A,B$ with $P(B)>0$,
$$P(A\mid B)=\frac{P(A\cap B)}{P(B)}.$$
Conditioning restricts the sample space to $B$ and renormalises. Consequences:
- **Product (chain) rule:** $P(A\cap B)=P(A\mid B)P(B)$, and for many variables $P(x_1,\dots,x_n)=\prod_i P(x_i\mid x_1,\dots,x_{i-1})$.
- **Law of total probability:** if $\{B_i\}$ partitions the sample space, $P(A)=\sum_i P(A\mid B_i)P(B_i)$.
- **Bayes' theorem:** $P(B\mid A)=P(A\mid B)P(B)/P(A)$; see [[bayes_theorem]].
- **Independence:** $A$ and $B$ are independent exactly when $P(A\mid B)=P(A)$. Conditional independence $A\perp B\mid C$ means $P(A\mid B,C)=P(A\mid C)$ ([[conditional_independence]]).

Two cautions. $P(A\mid B)$ is generally not equal to $P(B\mid A)$; confusing them is the prosecutor's fallacy. And for continuous variables the event $\{X=x\}$ has probability zero, so conditioning is defined through densities, $p(y\mid x)=p(x,y)/p(x)$, or more generally through regular conditional distributions. The result can depend on the parameterisation of the conditioning event, as in the Borel-Kolmogorov paradox.

A small example: with prior $P(s)=(0.5,0.5)$ and likelihood $P(o\mid s)=(0.8,0.3)$, the joint is $(0.40,0.15)$, the evidence is $0.55$, and $P(s\mid o)=(0.727,0.273)$.

## Role in Active Inference
Every array in a discrete Active Inference model is a conditional distribution. The likelihood matrix $A$ holds $P(o\mid s)$ with columns summing to one, the transition arrays $B(a)$ hold $P(s'\mid s,a)$, the preferences $C$ and prior $D$ define marginals, and the policy posterior is a distribution conditioned on beliefs about the future. Inference itself is conditioning the joint model on observations, and variational free energy is a bound on the surprise $-\ln P(o)$ obtained by marginalising these conditionals. Directed graphical models make the factorisation explicit ([[bayesian_networks]]).

## Related concepts
- [[probability_theory]] — measure-theoretic setting for conditioning
- [[bayes_theorem]] — inversion of a conditional
- [[conditional_independence]] — structure that simplifies conditionals
- [[bayesian_networks]] — joint distributions built from conditionals
- [[action_distribution]] — conditional distribution over actions
- [[probability_distributions]] — families used to parameterise conditionals
