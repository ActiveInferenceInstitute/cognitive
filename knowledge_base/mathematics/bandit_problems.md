---
title: Bandit Problems
description: The formal framework of sequential decision problems with partial feedback, covering stochastic, adversarial and contextual variants, regret as the performance measure, and the link to Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - decision_theory
  - regret
  - sequential_decision
semantic_relations:
  - type: relates
    links:
      - "[[exploration_exploitation]]"
      - "[[multi_armed_bandits]]"
      - "[[decision_theory]]"
      - "[[reinforcement_learning]]"
      - "[[epistemic_value]]"
---

# Bandit Problems

## Overview
A bandit problem is a sequential decision problem in which, at each round $t=1,\dots,T$, a learner selects an action $a_t$ from a set $\mathcal{A}$ and observes only the reward of the chosen action. Because unchosen actions reveal nothing, the learner must balance gathering information with collecting reward. Bandit problems are the simplest setting in which this exploration-exploitation dilemma arises, since the chosen action affects what is learned but, unlike in a Markov decision process, does not change any state of the world. The name comes from a gambler facing slot machines ("one-armed bandits"). Robbins (1952) is a standard early reference for the sequential design formulation.

## Problem classes and regret
- **Stochastic**: each action has a fixed reward distribution with mean $\mu_a$, and rewards are drawn independently.
- **Adversarial**: rewards are chosen by an adversary, and guarantees hold for any reward sequence.
- **Contextual**: a context $x_t$ is observed before acting, and the best action depends on it.
- **Structured or continuous-armed**: actions share structure, for example linear or Lipschitz rewards.

Performance is measured by regret against the best fixed action in hindsight. In the stochastic case, with $\mu^{*}=\max_a\mu_a$, the expected regret is $R_T=T\mu^{*}-\mathbb{E}\sum_{t=1}^{T}r_t$. Lai and Robbins (1985) showed that for broad classes of reward distributions no consistent strategy can have regret growing slower than logarithmically in $T$, and algorithms such as UCB1 attain this order (Auer et al., 2002).

## Relation to Active Inference
Bandit tasks are a common test bed for Active Inference. The agent holds beliefs about which arm is best, and expected free energy adds an epistemic term to the pragmatic term, so uncertainty about arm payoffs can drive sampling in a way that resembles the optimism bonus in UCB (the epistemic term is described in Friston et al., 2015). Empirical comparisons of such agents with standard bandit algorithms exist, with results depending on the task and parameter settings. The link is an applied modelling correspondence, not an equivalence: bandit theory is built around regret bounds, whereas Active Inference does not optimise regret.

## Related concepts
- [[multi_armed_bandits]] — the canonical finite-armed instance
- [[exploration_exploitation]] — the dilemma that bandits isolate
- [[decision_theory]] — the general setting for choice under uncertainty
- [[reinforcement_learning]] — bandits as the one-state special case
- [[epistemic_value]] — the information-seeking term in Active Inference

## References
- Robbins, H. (1952). Some aspects of the sequential design of experiments. Bulletin of the American Mathematical Society, 58, 527-535.
- Lai, T. L., & Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. Advances in Applied Mathematics, 6, 4-22.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47, 235-256.
- Lattimore, T., & Szepesvari, C. (2020). Bandit Algorithms. Cambridge University Press.
- Friston, K., Rigoli, F., Ognibene, D., Mathys, C., Fitzgerald, T., & Pezzulo, G. (2015). Active inference and epistemic value. Cognitive Neuroscience, 6, 187-214.
