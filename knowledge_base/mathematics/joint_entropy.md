---
title: Joint Entropy
description: The entropy of two or more random variables considered together, its bounds, chain rule and relation to mutual information, and where joint distributions of states and observations enter free energy.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_theory
  - entropy
  - joint_distribution
semantic_relations:
  - type: relates
    links:
      - "[[entropy]]"
      - "[[conditional_entropy]]"
      - "[[mutual_information]]"
      - "[[cross_entropy]]"
      - "[[information_theory]]"
---

# Joint Entropy

## Overview
The joint entropy of two discrete random variables $X$ and $Y$ with joint distribution $p(x,y)$ is
$$
H(X,Y)=-\sum_{x,y}p(x,y)\log p(x,y).
$$
It is the entropy of the pair treated as a single random variable, and it measures the total uncertainty about both variables. For $n$ variables, $H(X_1,\dots,X_n)$ is defined likewise from the joint distribution. Units are bits or nats depending on the base of the logarithm.

## Properties
- **Bounds**: $\max\{H(X),H(Y)\}\le H(X,Y)\le H(X)+H(Y)$. The upper bound is attained if and only if $X$ and $Y$ are independent.
- **Chain rule**: $H(X,Y)=H(X)+H(Y\mid X)=H(Y)+H(X\mid Y)$, and in general $H(X_1,\dots,X_n)=\sum_i H(X_i\mid X_1,\dots,X_{i-1})$.
- **Mutual information**: $I(X;Y)=H(X)+H(Y)-H(X,Y)$, so the dependence between variables is the amount by which the joint entropy falls short of the sum of the marginals.
- **Symmetry**: $H(X,Y)=H(Y,X)$.
- **Functions**: applying a deterministic function to the pair cannot increase its entropy, $H(g(X,Y))\le H(X,Y)$; if $Y=g(X)$ then $H(X,Y)=H(X)$.
- **Subadditivity and submodularity**: for sets of variables, entropy is non-decreasing, subadditive and submodular, which underlies information inequalities used in network coding.

For continuous variables, the joint differential entropy has the same algebraic form but can be negative and changes under reparameterisation of the variables.

## Relation to Active Inference
The joint distribution of hidden states and observations, $p(o,s)$, is the generative model of Active Inference, and its entropy and expected log-probability appear in the main quantities. Variational free energy can be written using the expected negative log of this joint distribution, $\mathbb{E}_q[-\ln p(o,s)]$, minus the entropy of $q$, and average surprise over observations is the entropy of the sensory marginal, which the free energy principle states that adaptive systems keep low. The decomposition $H(o,s)=H(s)+H(o\mid s)$ contains the conditional entropy $H(o\mid s)$, which is the ambiguity term of expected free energy. These are identities of information theory, not further empirical claims.

## Related concepts
- [[entropy]] — uncertainty of a single variable
- [[conditional_entropy]] — what remains after one variable is known
- [[mutual_information]] — the shared information given by marginals minus the joint
- [[cross_entropy]] — different quantity with similar notation, between two distributions
- [[information_theory]] — the framework in which these measures are defined

## References
- Shannon, C. E. (1948). A mathematical theory of communication. Bell System Technical Journal, 27, 379-423 and 623-656.
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
- MacKay, D. J. C. (2003). Information Theory, Inference, and Learning Algorithms. Cambridge University Press.
