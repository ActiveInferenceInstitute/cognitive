---
title: Algorithmic Information
description: Algorithmic information theory defines the information content of an individual object by the length of its shortest program, giving Kolmogorov complexity and links to compression, randomness and model selection.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - information_theory
  - kolmogorov_complexity
  - computation
semantic_relations:
  - type: relates
    links:
      - "[[information_theory]]"
      - "[[computational_complexity]]"
      - "[[entropy]]"
      - "[[complexity_measures]]"
---

# Algorithmic Information

## Overview
Shannon information concerns the average uncertainty of a random source. Algorithmic information theory, developed by Solomonoff, Kolmogorov and Chaitin, assigns information to a single finite string without reference to a distribution. The central quantity is Kolmogorov complexity, the length of the shortest program that makes a fixed universal computer output the string and halt.

## Definition and properties
For a universal prefix machine $U$, define
$$K_U(x)=\min\{|p| : U(p)=x\}.$$
- **Invariance.** For two universal machines $U,V$ there is a constant $c_{U,V}$, independent of $x$, with $|K_U(x)-K_V(x)|\le c_{U,V}$. Complexity is therefore defined up to an additive constant.
- **Incompressibility.** Most strings of length $n$ have $K(x)\ge n-O(1)$; there are fewer than $2^{n-k}$ programs shorter than $n-k$ bits, so at most that fraction of strings can be compressed by $k$ bits.
- **Uncomputability.** No algorithm computes $K(x)$ for all $x$. It can be approximated from above by running ever more programs, never certified from below.
- **Link to entropy.** For a computable distribution, the expected $K(x)$ equals Shannon entropy up to a term that depends only on the distribution's description, so the two notions agree on average.
- **Algorithmic probability.** Solomonoff's universal prior weights a hypothesis by $2^{-K}$, favouring short descriptions; it underlies universal sequence prediction and is incomputable.

## Role in Active Inference
The connection is conceptual rather than formal. Free energy decomposes into accuracy and complexity, where complexity is the KL divergence between posterior and prior. That is a statistical description length: coding the data with a model costs the bits needed to specify the posterior relative to the prior (the minimum description length idea), and it is computable for a chosen model class. Kolmogorov complexity would be the model-free limit of such a measure. Active Inference does not use it directly, because it cannot be calculated; practitioners use tractable surrogates such as the KL term, BIC or coding length under a fixed code. Claims that brains minimise algorithmic complexity should be read as a regulative analogy.

## Related concepts
- [[information_theory]] — the probabilistic counterpart based on entropy
- [[entropy]] — the average-case quantity that algorithmic complexity matches in expectation
- [[computational_complexity]] — resource-bounded versions of description length
- [[complexity_measures]] — practical proxies for model and data complexity
- [[information_criteria]] — penalised-likelihood surrogates for description length
- [[coding_theory]] — constructive codes whose lengths bound description length

## References
- Li, M. and Vitányi, P. (2019). *An Introduction to Kolmogorov Complexity and Its Applications*, 4th ed. Springer.
- Cover, T. M. and Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed. Wiley.
