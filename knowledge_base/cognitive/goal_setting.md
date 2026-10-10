---
title: Goal Setting
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - active_inference
  - goals
  - preferences
aliases:
  - goal_specification
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[prior_beliefs]]"
      - "[[preference_learning]]"
---

# Goal Setting

## Overview
In Active Inference a goal is not a separate object that the agent pursues. Goals are prior preferences: a distribution over outcomes (or states) that the agent expects to occupy, and behaviour that realizes the goal is whatever minimizes expected free energy relative to that distribution.

## Core claims
**Goals as C.** In `DiscreteGenerativeModel`, `C[o]` holds log preferences over observations and the preferred distribution is p*(o) = softmax(C). Setting a goal means writing C. The risk term KL[q(o|π) || p*(o)] then measures how far the outcomes predicted under a policy lie from the goal; the agent is drawn to policies whose predicted outcomes match it.

**Goals over states.** The homeostatic model in `homeostatic.py` takes a target distribution over states and derives observation preferences by pushing it through the likelihood: p*(o) = A·target, then C = ln p*(o). Its docstring notes why: the state and observation spaces generally differ in size, so reading a state target directly as an outcome preference is only valid when the sizes happen to coincide.

**Control modes.** `HomeostaticControl` scores each action by how close its predicted beliefs are to the target; `AdaptiveControl` mixes target tracking with a prediction-error term weighted by `exploration_weight`, and `update_parameters` lowers that weight as a stability index rises (clipped to 0.1 to 0.5). These are control heuristics layered on the model, not part of the canonical scheme.

**Distinct from an action prior.** The `goal_prior` argument of the dispatcher has the shape of the action set: it is a prior over actions that is combined with the policy energies, not an outcome preference. Confusing the two is easy given the name.

**Hierarchy.** In hierarchical models, a higher level can set the preferences or priors of a lower level, so a long-horizon goal constrains short-horizon behaviour (see [[hierarchical_inference]]).

## Role in Active Inference
Because a goal is a prior, the framework gives no account of why a given goal is adopted; it is stipulated by the modeller or learned (see [[preference_learning]]). It does explain why a goal-pursuing agent is not a dark-room seeker: preferences make some predictable outcomes costly (see [[dark_room_problem]]). Goals can also conflict with the epistemic term, and which wins is controlled by how sharp C is relative to the uncertainty in the model.

## Related concepts
- [[prior_beliefs]] — the family of priors that includes goals
- [[preference_learning]] — how goals can be acquired
- [[pragmatic_value]] — the value of reaching a goal
- [[action_planning]] — where goals are turned into plans
- [[policy_selection]] — choosing among goal-directed policies
- [[dark_room_problem]] — why goals prevent trivial minimization
