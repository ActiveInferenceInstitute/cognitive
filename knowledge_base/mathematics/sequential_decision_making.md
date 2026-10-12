---
title: Sequential Decision Making
description: Choosing a series of actions whose consequences unfold over time, covering Markov decision processes, Bellman optimality, partial observability, and the planning-as-inference treatment used in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - decision_theory
  - planning
  - control
semantic_relations:
  - type: relates
    links:
      - "[[markov_decision_process]]"
      - "[[dynamic_programming]]"
      - "[[reinforcement_learning]]"
      - "[[pomdp_framework]]"
      - "[[policy_selection]]"
      - "[[decision_making]]"
wikidata: Q55643366
---

# Sequential Decision Making

## Overview
In sequential decision making an agent chooses actions over time, and each choice changes the state of the world and thereby the options and rewards that follow. The task differs from a one-off choice because the value of an action depends on the actions that will follow it. Central issues are credit assignment, the balance of exploration and exploitation, and the length of the planning horizon.

## Formal frameworks
The standard model is the Markov decision process, with states $s$, actions $a$, transition probabilities $P(s'\mid s,a)$, a reward $R(s,a)$ and a discount factor $\gamma$. A policy $\pi$ maps states to actions or to distributions over actions. The optimal value function satisfies the Bellman equation,
$$V^*(s)=\max_a\Big[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V^*(s')\Big],$$
which dynamic programming solves for small problems. When the state is only partially observed, the problem becomes a partially observable Markov decision process in which the agent acts on a belief over states. Reinforcement learning addresses the case where the model is unknown. Planning can be closed-loop, with actions conditioned on observations, or open-loop, with a fixed sequence chosen in advance. The number of open-loop sequences grows exponentially with the horizon, which motivates pruning, tree search and approximate methods.

## Behaviour and neuroscience
Human choices in multi-step tasks have been analysed for signatures of model-based planning, which uses a learned model of transitions, and model-free habit, which caches action values. Dopaminergic prediction errors and prefrontal and striatal circuits have been associated with these computations, though the mapping is debated.

## Relation to Active Inference
Active Inference treats sequential decision making as inference over policies. In the discrete-time formulation each policy is a sequence of actions, scored by its expected free energy, and the posterior over policies is a softmax of the negative scores. Because expected free energy includes an information-seeking term as well as preference satisfaction, exploration arises without a separate bonus. This differs from Bellman optimality, which maximises expected cumulative reward, though in special cases the two coincide. Whether the resulting behaviour fits human data better than reinforcement-learning alternatives is an empirical question under study.

## Related concepts
- [[markov_decision_process]] — the standard formal model
- [[dynamic_programming]] — exact solution by backward recursion
- [[reinforcement_learning]] — learning policies from experience
- [[pomdp_framework]] — the partially observed setting used in Active Inference
- [[planning_as_inference]] — casting planning as probabilistic inference
- [[action_distribution]] — probabilities over the actions chosen

## References
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
- Puterman, M. L. (1994). Markov Decision Processes: Discrete Stochastic Dynamic Programming. Wiley.
- Daw, N. D., Gershman, S. J., Seymour, B., Dayan, P., & Dolan, R. J. (2011). Model-based influences on humans' choices and striatal prediction errors. Neuron, 69, 1204-1215.
