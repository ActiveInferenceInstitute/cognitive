---
title: Conditional Entropy
description: The expected remaining uncertainty about a random variable once another is known, its chain rule and relation to mutual information, and its role in the ambiguity term of expected free energy.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_theory
  - entropy
  - uncertainty
semantic_relations:
  - type: relates
    links:
      - "[[entropy]]"
      - "[[joint_entropy]]"
      - "[[mutual_information]]"
      - "[[cross_entropy]]"
      - "[[expected_free_energy]]"
---

# Conditional Entropy

## Overview
The conditional entropy $H(Y\mid X)$ is the average uncertainty that remains about $Y$ after $X$ is observed. For discrete variables with joint distribution $p(x,y)$,
$$
H(Y\mid X)=-\sum_{x,y}p(x,y)\log p(y\mid x)=\sum_x p(x)\,H(Y\mid X=x).
$$
It is an average over the entropies of the conditional distributions $p(y\mid x)$ and is measured in bits or nats depending on the logarithm base. Shannon (1948) introduced it alongside entropy for analysing communication channels, where $H(Y\mid X)$ quantifies noise.

## Properties
- **Chain rule**: $H(X,Y)=H(X)+H(Y\mid X)$, so $H(Y\mid X)=H(X,Y)-H(X)$.
- **Conditioning reduces entropy on average**: $0\le H(Y\mid X)\le H(Y)$, with equality on the right if and only if $X$ and $Y$ are independent. A particular observation $X=x$ can raise the entropy of $Y$, but the average cannot.
- **Link to mutual information**: $I(X;Y)=H(Y)-H(Y\mid X)=H(X)-H(X\mid Y)$.
- **Determinism**: $H(Y\mid X)=0$ if and only if $Y$ is a function of $X$.
- **Asymmetry**: in general $H(Y\mid X)\neq H(X\mid Y)$.

For continuous variables the analogous conditional differential entropy can be negative and is not invariant under change of variables, so the discrete properties need care when transferred.

## Relation to Active Inference
The connection is a direct derivation. In the common decomposition of expected free energy $G(\pi)$ into risk and ambiguity, the ambiguity term is the expected conditional entropy of observations given hidden states, $\mathbb{E}_{q(s\mid\pi)}\,H[p(o\mid s)]$. A policy leading to states whose observations are noisy or unreliable has high ambiguity and is penalised, so the agent prefers states that provide precise sensory evidence. In the free energy principle literature, entropies of sensory states are central to the claim that adaptive systems bound the surprise of their observations; conditional entropy is the quantity that the likelihood mapping $A$ fixes for each state.

## Related concepts
- [[entropy]] — unconditional uncertainty of a single variable
- [[joint_entropy]] — uncertainty of two variables taken together
- [[mutual_information]] — the reduction in entropy produced by conditioning
- [[cross_entropy]] — entropy under a mismatched model
- [[expected_free_energy]] — contains the expected conditional entropy as ambiguity

## References
- Shannon, C. E. (1948). A mathematical theory of communication. Bell System Technical Journal, 27, 379-423 and 623-656.
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
