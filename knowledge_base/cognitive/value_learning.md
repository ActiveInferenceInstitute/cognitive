---
title: Value Learning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - active_inference
  - learning
  - value
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[preference_learning]]"
      - "[[reinforcement_learning]]"
---

# Value Learning

## Overview
Value learning is the acquisition of knowledge about what is worth pursuing. Active Inference has no reward function and no value function V(s) or Q(s, a); value is carried by prior preferences over outcomes and by the expected free energy of policies, so "learning value" can mean several different updates.

## Core claims
**Where value sits.** Pragmatic value is the expected log preference E_q(o|π)[ln p*(o)], with p*(o) = softmax(C). Epistemic value is the expected information gain about hidden states. In this repo's convention G = risk + ambiguity, and both appear in G (see [[knowledge_base/mathematics/expected_free_energy]]). On randomly generated models the code's G equals −E_q[ln p*(o)] − I(s; o) to numerical precision, so pragmatic and epistemic value are two terms of one objective rather than separate reward channels.

**Three things that can be learned.**
1. *The model* (A, B, D) through Dirichlet concentration parameters, e.g. a ← a + η · o ⊗ q(s). Policy values change because G is computed from the model, even though C is untouched.
2. *The preferences* C itself, covered in [[preference_learning]]. This is philosophically delicate: if C is a prior that defines what the agent is, learning it is changing the agent's identity, not just its knowledge.
3. *Habits*, a prior over policies; see [[policy_learning]].

**Contrast with reinforcement learning.** RL learns a value function from scalar reward and bootstraps with temporal-difference errors. Active Inference can recover reward-seeking behaviour by setting C proportional to reward, and Sajid and colleagues (2021) compare the two formulations in detail; but the converse is not true, since an epistemic drive has no counterpart in plain expected-reward maximization.

## Role in Active Inference
In this repository, value is fixed by construction: `DiscreteGenerativeModel` validates `C` but contains no routine that updates `A`, `B` or `C` from experience. `InferenceConfig.learning_rate` blends prior and posterior beliefs inside the dispatcher's belief update; it is not a value-learning rate. The homeostatic controller's `update_parameters` adjusts an exploration weight from a stability index, a heuristic control rule rather than Dirichlet learning. The learning rules above are therefore theory, not a description of an implemented learner.

## Related concepts
- [[preference_learning]] — learning the C vector
- [[reinforcement_learning]] — the reward-based alternative
- [[pragmatic_value]] — the goal-seeking term
- [[epistemic_value]] — the information-seeking term
- [[learning_theory]] — general frameworks for learning
- [[goal_setting]] — how value becomes a target

## References
- Sajid, N., Ball, P. J., Parr, T. & Friston, K. J. (2021). Active inference: demystified and compared. *Neural Computation*.
