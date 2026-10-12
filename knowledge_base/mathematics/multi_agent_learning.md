---
title: Multi-Agent Learning
description: Learning when several adaptive agents share an environment, formalised as stochastic games, with the problems of non-stationarity, equilibrium selection and credit assignment, and links to game theory.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - reinforcement_learning
  - game_theory
  - multi_agent_systems
semantic_relations:
  - type: relates
    links:
      - "[[game_theory]]"
      - "[[reinforcement_learning]]"
      - "[[population_games]]"
      - "[[replicator_dynamics]]"
      - "[[swarm_intelligence]]"
---

# Multi-Agent Learning

## Overview
Multi-agent learning studies how several agents improve their behaviour through experience while each agent's actions affect the others. The standard formal model is the Markov or stochastic game (Shapley, 1953), a tuple $(N,S,\{A_i\},P,\{r_i\},\gamma)$ of agents, states, per-agent action sets, a transition kernel $P(s'\mid s,a_1,\dots,a_N)$ over joint actions, per-agent rewards and a discount factor. A single-agent Markov decision process is the case $N=1$, and a repeated matrix game is the case with one state.

## Central difficulties
- **Non-stationarity.** From one agent's viewpoint the other learners are part of the environment, and their policies change as they learn. The convergence guarantees of single-agent methods, which assume a fixed transition kernel, therefore no longer apply.
- **Equilibrium selection.** Solution concepts such as Nash equilibrium need not be unique, and a learning process may cycle or settle on a poor equilibrium. In zero-sum games a minimax value is well defined (Littman, 1994); in general-sum games no comparably simple target exists.
- **Credit assignment.** In cooperative settings with a shared team reward it is hard to determine which agent's action produced a result.
- **Partial observability and communication.** Agents may see only local observations, making the problem decentralised.

Approaches include independent learners that ignore the others, joint-action learners that model other agents' policies, fictitious play and its smooth variants, and centralised training with decentralised execution. Self-play, in which an agent learns against copies of itself, has produced strong play in some two-player games. The surveys by Busoniu and colleagues (2008) and the textbook by Shoham and Leyton-Brown (2009) organise this literature.

## Relation to Active Inference
Evolutionary game theory links learning to population dynamics: replicator dynamics describe how strategy frequencies change under payoff-driven selection (see [[replicator_dynamics]]). Within Active Inference, multi-agent settings are treated by giving each agent a generative model that includes beliefs about the others, as in theory-of-mind and joint-action models, and by studying collective behaviour of coupled agents sharing a Markov blanket structure. The link is an active research programme and not a settled framework, and its empirical benefit over reinforcement-learning baselines is still being assessed.

## Related concepts
- [[game_theory]] — equilibrium concepts and payoff structure
- [[reinforcement_learning]] — the single-agent setting being generalised
- [[population_games]] — large-population limit of strategic interaction
- [[replicator_dynamics]] — evolutionary form of learning
- [[exploration_exploitation]] — a tradeoff that becomes harder with other learners
- [[swarm_intelligence]] — collective behaviour from simple interacting agents

## References
- Shapley, L. S. (1953). Stochastic games. Proceedings of the National Academy of Sciences, 39, 1095-1100.
- Littman, M. L. (1994). Markov games as a framework for multi-agent reinforcement learning. Proceedings of the 11th International Conference on Machine Learning, 157-163.
- Busoniu, L., Babuska, R., & De Schutter, B. (2008). A comprehensive survey of multiagent reinforcement learning. IEEE Transactions on Systems, Man, and Cybernetics, Part C, 38, 156-172.
- Shoham, Y., & Leyton-Brown, K. (2009). Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations. Cambridge University Press.
