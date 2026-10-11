---
title: Multi-Armed Bandits
description: The finite-armed bandit setting and its main strategies, including epsilon-greedy, upper confidence bound and Thompson sampling, with notes on how Active Inference agents are tested on bandit tasks.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - bandits
  - exploration
  - algorithms
semantic_relations:
  - type: relates
    links:
      - "[[bandit_problems]]"
      - "[[epsilon_greedy]]"
      - "[[exploration_methods]]"
      - "[[exploration_exploitation]]"
      - "[[acquisition_functions]]"
---

# Multi-Armed Bandits

## Overview
A multi-armed bandit has $K$ arms, each producing rewards from an unknown distribution. At every round the learner pulls one arm and sees its reward. Pulling the arm that currently looks best exploits existing knowledge, while pulling a less-tried arm explores. The general formulation, regret criterion and problem variants are covered under [[bandit_problems]]; this page concerns the finite-armed case and the algorithms used for it.

## Main strategies
- **Epsilon-greedy**: with probability $1-\varepsilon$ choose the arm with highest estimated mean, otherwise choose uniformly at random. It is simple, but a fixed $\varepsilon$ gives regret growing linearly in $T$ unless $\varepsilon$ is decayed.
- **Upper confidence bound (UCB1)**: choose the arm maximising $\hat\mu_a+\sqrt{2\ln t/n_a}$, where $n_a$ counts pulls of arm $a$. For rewards in $[0,1]$ this has regret of order $\ln T$ (Auer et al., 2002).
- **Thompson sampling**: keep a posterior over each arm's parameters, sample from it, and pull the arm with the best sampled value. It dates to Thompson (1933) and is competitive in practice.
- **Gittins index**: for the discounted Bayesian formulation with independent arms, an index policy is optimal, which reduces a joint problem over all arms to a computation for each arm separately.

Estimates of arm means are typically updated incrementally, $\hat\mu_a\leftarrow\hat\mu_a+(r-\hat\mu_a)/n_a$. When arms are correlated or have continuous parameters, the same ideas appear as acquisition functions in Bayesian optimization.

## Relation to Active Inference
Active Inference agents have been applied to bandit tasks, with arms represented as the contexts or states of a hidden-state factor and reward-related outcomes encoded as preferences. Expected free energy supplies an information-seeking term that favours uncertain arms, so exploration emerges from the same objective as exploitation, without a separate exploration parameter (the epistemic term is described in Friston et al., 2015). Reported comparisons with UCB and Thompson sampling depend on the task design; the correspondence is empirical and by analogy, since Active Inference does not carry the regret guarantees of the algorithms above.

## Related concepts
- [[bandit_problems]] — general formulation and regret
- [[epsilon_greedy]] — the simplest exploration rule
- [[exploration_methods]] — wider catalogue of exploration strategies
- [[exploration_exploitation]] — the underlying trade-off
- [[acquisition_functions]] — bandit-style scoring for continuous inputs

## References
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25, 285-294.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47, 235-256.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
- Friston, K., Rigoli, F., Ognibene, D., Mathys, C., Fitzgerald, T., & Pezzulo, G. (2015). Active inference and epistemic value. Cognitive Neuroscience, 6, 187-214.
