---
title: Markov Decision Process
description: The standard model of sequential decision making under uncertainty, with states, actions, transitions and rewards, its Bellman equations, and how it relates to the partially observed models used in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - decision_theory
  - sequential_decision_making
  - dynamic_programming
semantic_relations:
  - type: relates
    links:
      - "[[pomdp_framework]]"
      - "[[reinforcement_learning]]"
      - "[[value_function]]"
      - "[[active_inference_pomdp]]"
      - "[[transition_model]]"
      - "[[action_space]]"
wikidata: Q176789
---

# Markov Decision Process

## Overview
A Markov decision process (MDP) models an agent choosing actions in an environment whose state evolves stochastically. It is defined by a tuple $(\mathcal{S},\mathcal{A},P,R,\gamma)$: a state space, an action space, transition probabilities $P(s'\mid s,a)$, a reward function $R(s,a)$ and a discount factor $\gamma\in[0,1)$. The Markov property states that the next state depends on the past only through the current state and action. An MDP is the formal basis of [[reinforcement_learning]] and of much of stochastic control.

## Policies and value
A policy $\pi(a\mid s)$ specifies the agent's behaviour. Its value is the expected discounted return,
$$V^\pi(s)=\mathbb{E}_\pi\Big[\sum_{t\ge0}\gamma^t R(s_t,a_t)\,\Big|\,s_0=s\Big].$$
The optimal value function satisfies the Bellman optimality equation
$$V^*(s)=\max_a\Big[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V^*(s')\Big].$$
For finite discounted problems an optimal deterministic stationary policy exists. Value iteration and policy iteration compute it; the Bellman operator is a contraction with modulus $\gamma$ in the supremum norm, so repeated application converges to a unique fixed point (see [[value_function]]).

## Partial observability
When the agent observes only a noisy function of the state, the problem becomes a partially observable MDP (POMDP). The belief, a posterior over hidden states, is a sufficient statistic, so a POMDP can be rewritten as an MDP over beliefs, at the cost of a continuous state space (see [[pomdp_framework]]).

## Relation to Active Inference
Discrete-time Active Inference models are POMDPs: hidden states, observations and control states are linked by likelihood and transition arrays, with the $B$ arrays playing the role of $P(s'\mid s,a)$ (see [[active_inference_pomdp]]). The two frameworks differ in objective. An MDP agent maximises expected reward, while an Active Inference agent selects policies by expected free energy, which combines expected preference satisfaction with information gain about hidden states. How far reward maximisation can be recovered as a special case of Active Inference is still debated.

## Related concepts
- [[pomdp_framework]] — the partially observed extension
- [[reinforcement_learning]] — learning optimal policies in an MDP
- [[value_function]] — expected return and the Bellman equations
- [[active_inference_pomdp]] — the same structure under a free-energy objective
- [[transition_model]] — the $P(s'\mid s,a)$ component

## References
- Puterman, M. L. (1994). Markov Decision Processes: Discrete Stochastic Dynamic Programming. Wiley.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
- Bellman, R. (1957). Dynamic Programming. Princeton University Press.
