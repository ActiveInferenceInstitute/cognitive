---
title: Upper Confidence Bound Algorithms
description: Bandit algorithms that choose the action with the highest optimistic estimate of reward, the UCB1 index and its logarithmic regret bound, and comparisons with directed exploration in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - bandits
  - exploration
  - online_learning
semantic_relations:
  - type: relates
    links:
      - "[[multi_armed_bandits]]"
      - "[[regret_minimization]]"
      - "[[exploration_exploitation]]"
      - "[[bandit_problems]]"
      - "[[acquisition_functions]]"
---

# Upper Confidence Bound Algorithms

## Overview
Upper confidence bound (UCB) algorithms address the multi-armed bandit problem by the principle of optimism in the face of uncertainty. For each arm they compute an upper bound on its plausible mean reward, which is large when the arm has either a high empirical mean or little data, and they pull the arm with the highest bound. Exploration then follows automatically: an arm that has been tried rarely keeps a wide confidence interval until it has been sampled enough to be dismissed ([[exploration_exploitation]]).

## UCB1
Suppose there are $K$ arms with rewards in $[0,1]$. After $n$ total pulls, let arm $j$ have been pulled $n_j$ times with empirical mean $\bar x_j$. The UCB1 rule of Auer, Cesa-Bianchi and Fischer (2002) plays each arm once and then chooses
$$j_n=\arg\max_j\Big(\bar x_j+\sqrt{\frac{2\ln n}{n_j}}\Big).$$
The bonus term comes from Hoeffding's inequality and shrinks as $n_j$ grows, but increases slowly with $n$, so that no arm is abandoned permanently. Writing $\Delta_i=\mu^*-\mu_i$ for the gap between the best mean and that of arm $i$, the expected regret after $n$ pulls satisfies
$$\mathbb{E}[R_n]\le8\sum_{i:\Delta_i>0}\frac{\ln n}{\Delta_i}+\Big(1+\frac{\pi^2}{3}\Big)\sum_{j=1}^K\Delta_j.$$
This logarithmic growth is of the same order as the lower bound of Lai and Robbins (1985), although the constants are not tight ([[regret_minimization]]).

## Variants
- **Tuned and KL-based indices.** UCB-V uses empirical variances, and KL-UCB replaces Hoeffding bounds with Kullback-Leibler confidence sets that give sharper regret constants.
- **Bayesian UCB.** Posterior quantiles replace frequentist bounds.
- **Linear and contextual bandits.** LinUCB-type methods build ellipsoidal confidence sets for a linear reward model.
- **Gaussian-process bandits and optimisation.** The GP-UCB acquisition function adds a multiple of the predictive standard deviation to the posterior mean ([[acquisition_functions]], [[bayesian_optimization]]).


## Relation to Active Inference
Both approaches turn uncertainty into directed exploration, but through different mechanisms. UCB adds a confidence bonus to an estimate of reward. In Active Inference, the expected free energy of a policy contains an epistemic term, expected information gain, alongside a pragmatic term for preferred outcomes ([[epistemic_value]], [[expected_free_energy]]). Sajid et al. (2021) compare Active Inference with reinforcement learning on bandit-like tasks. The correspondence is heuristic, and the two schemes make different predictions and have different guarantees: UCB has finite-time regret bounds, whereas comparable bounds for expected free energy minimisation are not established in general.

## Related concepts
- [[multi_armed_bandits]] — the problem UCB solves
- [[regret_minimization]] — the performance criterion
- [[exploration_exploitation]] — the trade-off handled by optimism
- [[epsilon_greedy]] — undirected exploration, for contrast
- [[acquisition_functions]] — the UCB form in Bayesian optimisation
- [[epistemic_value]] — the information-seeking term in Active Inference

## References
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47, 235-256.
- Lai, T. L., & Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. Advances in Applied Mathematics, 6, 4-22.
- Lattimore, T., & Szepesvari, C. (2020). Bandit Algorithms. Cambridge University Press.
- Sajid, N., Ball, P. J., Parr, T., & Friston, K. J. (2021). Active inference: demystified and compared. Neural Computation, 33, 674-712.
