---
title: Search Strategies
type: guide
status: stable
created: 2025-03-03
updated: 2026-10-09
tags:
- search
- planning
semantic_relations:
  - type: relates_to
    links:
      - "[[policy_search_methods]]"
      - "[[policy_selection]]"
      - "[[expected_free_energy]]"
---


# Search Strategies

Planning in discrete Active Inference means finding action sequences with low expected free energy. With $N_a$ actions and a horizon of $H$ steps there are $N_a^{H}$ sequences, so the question is how much of that tree to visit and in what order.

## The search space

A policy $\pi = (a_1, \dots, a_H)$ is scored by $G(\pi) = \sum_{\tau} \delta^{\tau-1}\, G(\pi, \tau)$, where $\delta$ is the discount factor and each term is the one-step risk plus ambiguity evaluated on the belief predicted under the actions so far. Two structural facts shape the search. First, the one-step cost depends on the whole prefix through the predicted belief, so equal-looking prefixes cannot be merged and exact dynamic programming over beliefs is not available. Second, the cost is additive across steps, so a partial sum is a lower bound on the final cost whenever one-step costs are non-negative. They are under the default `exploration_weight` of 0, since risk is a KL divergence and ambiguity an entropy; a positive extra epistemic weight can make a step negative and breaks the bound.

## Strategies

**Exhaustive enumeration.** Visit every sequence. Exact, and the right choice when $N_a^{H}$ is a few thousand. `DiscreteGenerativeModel.enumerate_policies(horizon, limit)` yields sequences in lexicographic order, lazily. With 2 actions and horizon 10 there are 1024 sequences.

**Breadth-first expansion with pruning (beam search).** Expand all actions one step, keep the $w$ best partial sequences by accumulated $G$, repeat. Cost is about $H\,w\,N_a$ one-step evaluations instead of $N_a^{H}\,H$. It is greedy at width 1 and can discard a prefix that pays off later, so widen it when late rewards matter.

**Depth-first with bounds (branch and bound).** Because partial sums only grow, any prefix whose cost already exceeds the best complete policy can be abandoned. This keeps exactness while skipping subtrees.

**Sampling.** Draw sequences at random, or adapt the sampling distribution toward good ones as the cross-entropy method does ([[policy_search_methods]]). Monte Carlo tree search belongs here too; this repository does not contain one.

**Heuristic restriction.** Restrict the policy set by design, for example to repeat-one-action policies, or use the action prior `E` to propose likely sequences.

## A concrete comparison

On the toy model of [[worked_example_efe]] (beliefs $(0.3, 0.7)$, horizon 10, undiscounted), exhaustive search over 1024 sequences finds a minimum of $3.6385$ at the first of 512 tied optimal sequences, `(1, 0, 0, 0, 0, 0, 0, 0, 0, 0)`. The ties arise because after the first move the belief $(0.9, 0.1)$ is unchanged by either action. Beam search of width 1, 2 and 4 returns the same cost. This problem does not stress beam search, since the optimum is also greedy-optimal; it only shows the interface and the cost difference.

## Limits in the dispatcher

`InferenceConfig.policy_limit` (default 4096) truncates enumeration in lexicographic order. All sequences beginning with action 0 come first, so a low limit can leave later first actions unevaluated. `calculate_expected_free_energy` detects this and raises a `ValueError` rather than silently assigning those actions zero probability. With 2 actions and horizon 13 there are 8192 sequences, the first 4096 all begin with action 0, and the error is raised. Raise the limit, reduce the horizon, or replace enumeration with a sampled search.

## Reproducing the numbers

```python
import numpy as np

from cognitive.models.active_inference import (
    ActiveInferenceDispatcher, DiscreteGenerativeModel, InferenceConfig, ModelState)

A = np.array([[0.9, 0.5], [0.1, 0.5]])
B = np.zeros((2, 2, 2))
B[:, :, 0] = np.eye(2)
B[:, :, 1] = [[0.9, 0.9], [0.1, 0.1]]
model = DiscreteGenerativeModel(A, B, np.array([2.0, 0.0]), np.full(2, 0.5), np.full(2, 0.5))
beliefs = np.array([0.3, 0.7])
horizon = 10

# Exhaustive: 2**10 = 1024 sequences, each scored over 10 steps.
policies = model.enumerate_policies(horizon)
scores = [model.evaluate_policy(beliefs, p, discount_factor=1.0) for p in policies]
best = int(np.argmin(scores))
print(len(policies), policies[best], round(scores[best], 4))


def beam_search(width):
    beams = [((), beliefs, 0.0)]
    for _ in range(horizon):
        candidates = []
        for sequence, belief, total in beams:
            for action in range(model.num_actions):
                g = model.expected_free_energy(belief, action)[0]
                candidates.append((sequence + (action,), model.predict_states(belief, action), total + g))
        beams = sorted(candidates, key=lambda c: c[2])[:width]
    return beams[0]


for width in (1, 2, 4):
    sequence, _, total = beam_search(width)
    print("beam", width, sequence, round(total, 4))

# policy_limit truncates in lexicographic order, so a low limit starves later first actions.
config = InferenceConfig(method="variational", policy_type="discrete", temporal_horizon=13,
                         learning_rate=0.5, precision_init=1.0, policy_limit=4096)
dispatcher = ActiveInferenceDispatcher(config, model)
state = ModelState(beliefs=beliefs, policies=np.full(2, 0.5), precision=1.0,
                   free_energy=0.0, prediction_error=0.0)
try:
    dispatcher.calculate_expected_free_energy(state)
except ValueError as error:
    print("ValueError:", str(error)[:60])
```

## Links

- [[optimization_strategies]]
- [[exploration_methods]]
- [[policy_search_methods]] · [[policy_selection]] · [[expected_free_energy]]
