---
title: Action Planning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - active_inference
  - planning
  - policy
aliases:
  - motor_planning
  - planning
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[planning_as_inference]]"
      - "[[policy_selection]]"
---

# Action Planning

## Overview
Action planning is choosing a sequence of actions by predicting its consequences. In Active Inference, planning is inference: candidate policies are scored by the expected free energy of the futures they are predicted to produce, and the agent acts on the resulting distribution over policies.

## Core claims
**Algorithm as implemented here.** The discrete scheme in `code/tools/src/models/active_inference/` has four steps.
1. *Enumerate.* `enumerate_policies(horizon, limit)` lists action sequences in lexicographic order via a product over the action set, truncated lazily at `limit` (`InferenceConfig.policy_limit`, default 4096).
2. *Roll forward.* For each policy, current state beliefs are propagated through `B[:, :, a]`, giving predicted states and predicted observations A·q(s) at each step.
3. *Score.* Each step contributes G = risk + ambiguity, where risk is KL[q(o) || softmax(C)] and ambiguity is E_q(s)[H[P(o|s)]]; steps are discounted by `discount_factor**step`, and the policy score is the sum.
4. *Aggregate.* Policies sharing a first action are combined by `MIN` or `MARGINAL`, then softmax(−score/temperature) gives a distribution over first actions.

**Properties worth knowing.**
- Planning is open-loop: the rollout propagates a single predicted belief and does not branch on observations that might arrive mid-policy. Branching, tree-search variants are the idea behind sophisticated inference (Friston and colleagues, 2021) and this dispatcher does not provide them; it accepts only `PolicyType.DISCRETE`.
- Cost grows as `num_actions ** horizon`; three actions over a horizon of four gives 81 policies, ten actions over six gives a million, which is why `policy_limit` exists. If the limit would leave some first action with no policy at all, the dispatcher raises instead of silently assigning it zero probability.
- Epistemic behaviour is built into the score through ambiguity, so planning to gather information needs no extra bonus; `exploration_weight` (0 to 1) subtracts an additional multiple of expected information gain.

## Role in Active Inference
Planning is where preferences (`C`), beliefs about dynamics (`B`), beliefs about observations (`A`) and the action prior (`E`) all meet. It is also the main source of computational cost in the framework, and the main place where approximations (restricted horizons, tree search, amortization) enter. The result is only as good as the model: a wrong `B` produces coherent but ineffective plans.

## Related concepts
- [[planning_as_inference]] — the general formulation
- [[policy_selection]] — turning policy scores into an action
- [[policy_evaluation]] — scoring individual policies
- [[knowledge_base/mathematics/expected_free_energy]] — the score being summed
- [[decision_making]] — the broader decision-theoretic setting
- [[exploration_exploitation]] — the epistemic/pragmatic trade-off inside the score
- [[goal_setting]] — where the preferences used in planning come from

## References
- Friston, K., Da Costa, L., Hafner, D., Hesp, C. & Parr, T. (2021). Sophisticated inference. *Neural Computation*.
