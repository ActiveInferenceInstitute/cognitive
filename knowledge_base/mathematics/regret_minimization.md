---
title: Regret Minimization
description: The online-learning objective of keeping cumulative regret against the best fixed action sublinear, with bounds for experts and bandit problems and connections to game-theoretic equilibria.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - online_learning
  - bandits
  - decision_theory
semantic_relations:
  - type: relates
    links:
      - "[[exploration_exploitation]]"
      - "[[multi_armed_bandits]]"
      - "[[ucb_algorithms]]"
      - "[[game_theory]]"
      - "[[decision_theory]]"
---

# Regret Minimization

## Overview
Regret minimization is an objective for sequential decision-making under uncertainty in which an algorithm is judged by how much worse it does than a benchmark chosen in hindsight. Over $T$ rounds, with reward $r_t(a)$ for action $a$, the cumulative regret relative to the best fixed action is
$$R_T=\max_{a}\sum_{t=1}^Tr_t(a)-\sum_{t=1}^Tr_t(a_t).$$
An algorithm has no regret if $R_T/T\to0$, meaning its average performance approaches that of the best fixed action. The objective needs no probabilistic model of the environment, so the guarantees can hold even against adversarially chosen rewards.

## Standard results
- **Full information (experts).** With $N$ actions and all rewards revealed after each round, multiplicative-weights (Hedge) algorithms achieve $R_T=O(\sqrt{T\ln N})$ for rewards in $[0,1]$, and this rate cannot be improved in general (Cesa-Bianchi & Lugosi, 2006).
- **Adversarial bandits.** If only the chosen action's reward is seen, the EXP3 algorithm attains $O(\sqrt{TK\ln K})$ for $K$ actions, in expectation (Auer et al., 2002).
- **Stochastic bandits.** When rewards for each arm are drawn from a fixed distribution, regret can be kept to logarithmic growth in $T$. Lai and Robbins (1985) proved an asymptotic lower bound on this logarithmic rate for suboptimal arms in terms of the Kullback-Leibler divergence between arm distributions, and upper-confidence-bound algorithms match its order ([[ucb_algorithms]]).
- **Games.** If every player in a repeated game uses a no-regret algorithm, the empirical distribution of joint play converges to a set of coarse correlated equilibria. In two-player zero-sum games, the average strategies converge to a minimax solution ([[game_theory]]).

Regret focuses on the trade-off between acquiring information and using it: low regret requires trying uncertain options enough to find the best one without paying too much for the search ([[exploration_exploitation]], [[multi_armed_bandits]]).

## Relation to Active Inference
The two frameworks pursue different objectives. Regret minimization evaluates an algorithm against a hindsight benchmark with worst-case guarantees. Active Inference agents minimise expected free energy, which combines expected utility with information gain, so exploration emerges from the same objective as exploitation. Comparative studies of Active Inference and reinforcement learning on bandit tasks exist, and they typically report empirical performance instead of regret bounds. Whether expected free energy minimisation has favourable regret properties is a question for specific models, with no general theorem. See [[expected_free_energy]] and [[epistemic_value]].

## Related concepts
- [[exploration_exploitation]] — the trade-off that regret quantifies
- [[multi_armed_bandits]] — the canonical setting for regret bounds
- [[ucb_algorithms]] — optimism-based algorithms with logarithmic regret
- [[bandit_problems]] — formal statements of the problem
- [[game_theory]] — regret and equilibrium
- [[expected_free_energy]] — the Active Inference alternative objective

## References
- Lai, T. L., & Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. Advances in Applied Mathematics, 6, 4-22.
- Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002). The nonstochastic multiarmed bandit problem. SIAM Journal on Computing, 32, 48-77.
- Cesa-Bianchi, N., & Lugosi, G. (2006). Prediction, Learning, and Games. Cambridge University Press.
