---
title: Action Space
description: The set of actions an agent may take in a decision problem, covering discrete and continuous forms, policy counts, and how action spaces enter POMDPs and Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - decision_theory
  - pomdp
  - policies
semantic_relations:
  - type: relates
    links:
      - "[[pomdp_framework]]"
      - "[[policy_selection]]"
      - "[[action_selection]]"
      - "[[state_space]]"
---

# Action Space

## Overview
An action space is the set $\mathcal{A}$ of choices available to an agent at a decision point. It is one component of a decision problem, alongside the state space, the observation space, the dynamics and a preference or reward structure. In Active Inference the term is used in two senses: a finite set of control states indexing the transition matrices of a discrete model, and a continuous space of effector variables acting on the world in continuous-time models.

## Definition and structure
- **Discrete.** $\mathcal{A}=\{a_1,\dots,a_K\}$, for example {up, down, left, right, stay} in a grid world. Each action selects one transition matrix $B(a)$, whose columns are distributions over next states.
- **Continuous.** $\mathcal{A}\subseteq\mathbb{R}^m$, for example joint torques or muscle activations. Actions enter the generative process directly, and the agent's actions change sensory input through $\partial o/\partial a$.
- **Factorised.** Several controllable factors give $\mathcal{A}=\mathcal{A}_1\times\dots\times\mathcal{A}_F$, with a separate $B$ array per factor.
- **State dependence.** In some problems the feasible set depends on the state, $\mathcal{A}(s)$. Forbidden moves are usually encoded by making them leave the state unchanged, or by a very low prior probability.

A **policy** is a sequence of actions $\pi=(a_t,\dots,a_{T-1})$. With $K$ actions and horizon $T$ there are $K^{T}$ open-loop policies, so four actions over three steps give 64. This exponential growth is the main reason planning is restricted by short horizons, pruning, tree search or learned habit priors.

## Role in Active Inference
In discrete-time Active Inference (the POMDP formulation) actions are not chosen by maximising a reward. A prior over policies is formed from the negative expected free energy of each policy, passed through a softmax, and an action is sampled from the resulting posterior. The size of $\mathcal{A}$ therefore fixes the number of policies whose expected free energy must be evaluated; see [[expected_free_energy]] and [[policy_selection]]. The prior over policies, often written $E$, is stored in the E matrix, which is defined over this space. In continuous-time formulations the action is a state variable updated by gradient descent on variational free energy, with the world's response supplying the gradient. The action space is therefore part of the agent's problem specification, not something inferred, although structure learning work asks how to enlarge it.

## Related concepts
- [[pomdp_framework]] — the decision model in which the action space is defined
- [[policy_selection]] — how sequences drawn from the action space are scored and chosen
- [[action_selection]] — mapping from policy posteriors to a single action
- [[action_distribution]] — probability distribution over actions
- [[state_space]] — the companion space of hidden states
- [[expected_free_energy]] — the quantity used to score policies
