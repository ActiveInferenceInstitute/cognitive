---
title: Statistical Complexity
description: The computational-mechanics measure of a process's structure, defined as the Shannon entropy of its causal states, and how it differs from randomness, excess entropy and the complexity term of free energy.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - computational_mechanics
  - complexity_measures
  - information_theory
semantic_relations:
  - type: relates
    links:
      - "[[complexity_measures]]"
      - "[[entropy]]"
      - "[[mutual_information]]"
      - "[[information_dynamics]]"
      - "[[stochastic_processes]]"
      - "[[markov_blanket]]"
---

# Statistical Complexity

## Overview
Statistical complexity, introduced by Crutchfield and Young (1989) within computational mechanics, quantifies how much structure a stationary stochastic process has, measured as the amount of information a model must store to predict it optimally. It is low both for perfectly ordered and for completely random processes, and highest for processes with intermediate, structured organisation.

## Definition
Let $\overleftarrow{X}$ be the semi-infinite past and $\overrightarrow{X}$ the future of a stationary process. Two pasts are causally equivalent if they induce the same conditional distribution over futures:
$$\overleftarrow{x}\sim\overleftarrow{x}'\iff P(\overrightarrow{X}\mid\overleftarrow{x})=P(\overrightarrow{X}\mid\overleftarrow{x}').$$
The equivalence classes are the causal states $\sigma\in\mathcal{S}$. Together with the transitions between them they form the $\varepsilon$-machine, the minimal unifilar predictor of the process. The statistical complexity is the entropy of the causal-state distribution,
$$C_\mu = H[\mathcal{S}] = -\sum_{\sigma\in\mathcal{S}}P(\sigma)\log P(\sigma).$$

## Properties
- A fair coin flipped independently has one causal state, so $C_\mu=0$. A constant sequence also has one causal state and $C_\mu=0$. A periodic sequence of period $p$ has $p$ equiprobable states and $C_\mu=\log p$.
- The excess entropy $E=I(\overleftarrow{X};\overrightarrow{X})$, the mutual information between past and future, satisfies $E\le C_\mu$. The gap measures how much stored structure is hidden from an observer of the output alone.
- Causal states are minimal sufficient statistics of the past for predicting the future, which makes $C_\mu$ a property of the process and not of a chosen model class.
- It differs from Kolmogorov-Chaitin complexity, which assigns high values to random strings.

## Relation to Active Inference
The connection to the free energy principle is conceptual. Internal states of a system with a [[markov_blanket]] are described as encoding beliefs about external states, which parallels causal states as sufficient statistics of the past. No theorem identifies the two. The word "complexity" also has a separate technical meaning in Active Inference, where variational free energy decomposes into accuracy and a complexity term equal to the KL divergence of the posterior from the prior; this should not be confused with $C_\mu$. The entry in [[free_energy_principle]] points here only for the comparison of structure measures.

## Related concepts
- [[complexity_measures]] — the wider family of quantities that this measure belongs to
- [[entropy]] — the quantity $C_\mu$ is built from
- [[mutual_information]] — used in the excess entropy bound
- [[information_dynamics]] — time-resolved information measures for processes
- [[stochastic_processes]] — the objects to which the definition applies

## References
- Crutchfield, J. P., & Young, K. (1989). Inferring statistical complexity. Physical Review Letters, 63, 105-108.
- Shalizi, C. R., & Crutchfield, J. P. (2001). Computational mechanics: Pattern and prediction, structure and simplicity. Journal of Statistical Physics, 104, 817-879.
- Grassberger, P. (1986). Toward a quantitative theory of self-generated complexity. International Journal of Theoretical Physics, 25, 907-938.
