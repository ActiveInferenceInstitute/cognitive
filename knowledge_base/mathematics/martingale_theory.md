---
title: Martingale Theory
description: Stochastic processes whose conditional expected future value equals the present value, with the optional stopping and convergence theorems and their uses in sequential inference and learning theory.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - probability
  - stochastic_processes
  - sequential_inference
semantic_relations:
  - type: relates
    links:
      - "[[stochastic_processes]]"
      - "[[probability_theory]]"
      - "[[conditional_probability]]"
      - "[[bayesian_inference]]"
      - "[[measure_theory]]"
---

# Martingale Theory

## Overview
A martingale is a stochastic process that models a fair game: given everything observed so far, the expected next value equals the current value. Formally, a process $(M_t)$ adapted to a filtration $(\mathcal{F}_t)$, which encodes the information available at time $t$, is a martingale if $\mathbb{E}|M_t|<\infty$ and

$$\mathbb{E}[M_{t+1}\mid\mathcal{F}_t]=M_t.$$

If the equality is replaced by $\le$ the process is a supermartingale, and by $\ge$ a submartingale. The theory was developed largely by Doob (1953), and is a core part of modern [[probability_theory]].

## Key results
- **Examples.** A simple symmetric random walk is a martingale. So is the likelihood ratio $L_t=\prod_{i\le t}q(x_i)/p(x_i)$ when the data are drawn from $p$, which is the basis of sequential testing.
- **Optional stopping.** Under suitable conditions, for instance a bounded stopping time $\tau$, $\mathbb{E}[M_\tau]=\mathbb{E}[M_0]$. This yields results such as gambler's-ruin probabilities, and shows that no stopping rule can turn a fair game into a favourable one.
- **Martingale convergence.** An $L^1$-bounded martingale converges almost surely to a limit. A uniformly integrable martingale also converges in $L^1$.
- **Maximal and concentration inequalities.** Doob's inequality bounds the probability that a martingale becomes large, and the Azuma-Hoeffding inequality gives exponential tail bounds for martingales with bounded increments, used widely in learning theory.
- **Doob decomposition.** Any adapted integrable process splits uniquely into a martingale part and a predictable part.

## Relation to Bayesian inference and Active Inference
The link is a structural property of Bayesian updating, not a specific claim about Active Inference. For a fixed event or parameter, the sequence of posterior probabilities $P(\theta\in A\mid x_{1:t})$ is a martingale under the agent's own predictive distribution: an ideal Bayesian cannot predict in which direction its beliefs will move. By the convergence theorem, posterior beliefs converge almost surely, which underlies consistency arguments for Bayesian learning. The same fact is sometimes used to formalise the idea that prediction errors of a calibrated generative model are unpredictable, so that precisely predicted sensory input gives no new information. Active Inference models rarely invoke martingale machinery explicitly; it appears mostly in the analysis of stochastic dynamics, sequential decision problems and regret bounds. Treating belief updating as a martingale presumes the model matches the data-generating process, which an actual organism need not satisfy.

## Related concepts
- [[stochastic_processes]] — the general class that martingales belong to
- [[probability_theory]] — foundations on which the definitions rest
- [[conditional_probability]] — conditional expectation defines the martingale property
- [[bayesian_inference]] — posterior probabilities form a martingale
- [[measure_theory]] — filtrations and conditional expectation are measure-theoretic

## References
- Doob, J. L. (1953). Stochastic Processes. Wiley.
- Williams, D. (1991). Probability with Martingales. Cambridge University Press.
