---
title: Excess Entropy
description: The mutual information between the infinite past and future of a stationary process, equal to the sum of entropy-rate convergence terms, used as a measure of memory and structure.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_theory
  - complexity
  - stochastic_processes
semantic_relations:
  - type: relates
    links:
      - "[[entropy]]"
      - "[[mutual_information]]"
      - "[[complexity_measures]]"
      - "[[information_dynamics]]"
      - "[[stochastic_processes]]"
---

# Excess Entropy

## Overview
Excess entropy measures how much of a stationary stochastic process's future is determined by its past. It quantifies the structure that remains after the irreducible randomness, the entropy rate, is accounted for. A process of independent coin flips has zero excess entropy, a perfectly periodic sequence has a finite positive value, and processes with long-range correlations can have large values.

## Definition
Let $H(L)$ be the Shannon entropy of a block of $L$ consecutive symbols and let $h_\mu=\lim_{L\to\infty}H(L)/L$ be the entropy rate. The excess entropy is
$$\mathbf{E}=\lim_{L\to\infty}\bigl[H(L)-h_\mu L\bigr],$$
when the limit exists. Equivalently, with $h_\mu(L)=H(L)-H(L-1)$ the block-conditional entropy,
$$\mathbf{E}=\sum_{L=1}^{\infty}\bigl[h_\mu(L)-h_\mu\bigr],$$
which is the total entropy that the finite-block estimates overstate. Crutchfield and Feldman (2003) show that $\mathbf{E}=I(\overleftarrow{X};\overrightarrow{X})$, the mutual information between the semi-infinite past and the semi-infinite future. Because it depends only on the process distribution, it is symmetric under time reversal.

## Names and related quantities
Grassberger (1986) introduced the quantity as effective measure complexity. Bialek, Nemenman and Tishby (2001) studied the same quantity as the predictive information, the information the past carries about the future, and related its growth with window length to the complexity of the model class needed for prediction. The statistical complexity of computational mechanics, the entropy of the causal states, is an upper bound on $\mathbf{E}$, so two processes can share an excess entropy yet require different amounts of stored memory.

## Relation to Active Inference
The connection is an analogy at the level of information theory. The free energy principle describes a system whose states are informative about external states through a Markov blanket, and measures such as predictive information have been used to ask how much of a system's dynamics predicts its future. No result derives excess entropy from the principle, and the quantity is not part of its standard formulation. It is better seen as a tool for characterising the temporal structure of data that generative models must capture.

## Related concepts
- [[mutual_information]] — the quantity of which excess entropy is a special case
- [[entropy]] — block entropies and entropy rate underlie the definition
- [[conditional_entropy]] — the block-conditional entropies used in the sum
- [[complexity_measures]] — excess entropy is one of several structural complexity measures
- [[information_dynamics]] — information-theoretic analysis of how processes unfold in time

## References
- Grassberger, P. (1986). Toward a quantitative theory of self-generated complexity. International Journal of Theoretical Physics, 25, 907-938.
- Bialek, W., Nemenman, I., & Tishby, N. (2001). Predictability, complexity, and learning. Neural Computation, 13, 2409-2463.
- Crutchfield, J. P., & Feldman, D. P. (2003). Regularities unseen, randomness observed: levels of entropy convergence. Chaos, 13, 25-54.
