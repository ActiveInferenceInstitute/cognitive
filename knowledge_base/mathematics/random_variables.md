---
title: Random Variables
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - probability
  - measure_theory
  - distributions
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/probability_theory]]"
      - "[[knowledge_base/mathematics/probability_distributions]]"
      - "[[knowledge_base/mathematics/measure_theory]]"
      - "[[knowledge_base/mathematics/conditional_independence]]"
---

# Random Variables

## Overview
A random variable is a quantity whose value is determined by the outcome of a random experiment. Formally it is a measurable function from a probability space to a value space; in practice it is described by its distribution. In graphical and POMDP models, hidden states, observations and actions are all random variables, and the model is a joint distribution over them.

## Definition / Formalism
Given a probability space $(\Omega,\mathcal F,P)$, a random variable is a measurable map $X:\Omega\to\mathcal X$. Its distribution is the push-forward $P_X(B)=P(X\in B)$.

- Discrete $X$: probability mass function $p(x)=P(X=x)$, with $\sum_xp(x)=1$.
- Continuous $X$ on $\mathbb R^d$: density $p(x)\ge0$ with $\int p=1$ and $P(X\in B)=\int_Bp$.
- Cumulative distribution function $F(x)=P(X\le x)$, defined for any real-valued $X$.

Expectation, variance and conditional expectation:

$$\mathbb E[g(X)]=\sum_xg(x)p(x)\;\text{or}\;\int g\,p,\qquad \mathrm{Var}(X)=\mathbb E[X^2]-\mathbb E[X]^2,$$

$$\mathbb E[X]=\mathbb E\big[\mathbb E[X\mid Y]\big],\qquad \mathrm{Var}(X)=\mathbb E[\mathrm{Var}(X\mid Y)]+\mathrm{Var}(\mathbb E[X\mid Y]).$$

Variables $X,Y$ are independent iff $p(x,y)=p(x)p(y)$, and conditionally independent given $Z$ iff $p(x,y\mid z)=p(x\mid z)p(y\mid z)$. For a smooth monotone map $Y=g(X)$ in one dimension the density transforms as $p_Y(y)=p_X(g^{-1}(y))\,|d g^{-1}/dy|$. A categorical variable on $n$ outcomes is represented by a probability vector in the simplex, or by a one-hot vector whose expectation is that probability vector.

## Role in Active Inference
Everything in a discrete active inference model is a random variable with a stated distribution. In this repo's `DiscreteGenerativeModel`, hidden state $s$ has prior $D$, observation $o$ has conditional distribution $A[o,s]=P(o\mid s)$ (each column of $A$ is a pmf), the next state has $B[s',s,a]=P(s'\mid s,a)$, and the action has prior $E$. Preferences $C$ are not probabilities but log-preferences, turned into a distribution over observations by a softmax. The factorisation $p(o,s)=p(o\mid s)p(s)$ and conditional independence structure are what make belief updating tractable. The law of total variance above is the basis for splitting predictive uncertainty into expected noise plus variation across states.

## Related concepts
- [[probability_theory]] — axioms and rules these objects obey
- [[probability_distributions]] — specific families and their parameters
- [[measure_theory]] — the measurable-function definition
- [[conditional_independence]] — structure that factorises the joint
- [[matrix_operations]] — how pmfs are stored as column-stochastic matrices
- [[uncertainty_quantification]] — variance and entropy as spread measures
