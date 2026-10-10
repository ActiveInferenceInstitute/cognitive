---
title: Basic API usage
type: guide
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[docs/api/api_reference]]"
      - "[[docs/examples/index]]"
---

# Basic API usage

Build a two-state model, filter a sequence of observations, and read the
policy distribution. Observations are integer indices into the rows of `A`.

```python
import numpy as np

from cognitive import (
    ActiveInferenceDispatcher,
    DiscreteGenerativeModel,
    InferenceConfig,
    ModelState,
)

model = DiscreteGenerativeModel(
    A=np.array([[0.9, 0.1], [0.1, 0.9]]),
    B=np.stack([np.eye(2), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2),
    C=np.array([0.0, 1.0]),
    D=np.array([0.5, 0.5]),
    E=np.array([0.5, 0.5]),
)
dispatcher = ActiveInferenceDispatcher(
    InferenceConfig("variational", "discrete", 2, 0.5, 1.0, seed=7), model
)

state = ModelState(model.D.copy(), model.E.copy(), 1.0, 0.0, 0.0)
for observation in (1, 1, 0):
    beliefs = dispatcher.dispatch_belief_update(observation, state)
    state = ModelState(beliefs, state.policies, state.precision, 0.0, 0.0)

policies = dispatcher.dispatch_policy_inference(state)
assert np.isclose(state.beliefs.sum(), 1.0) and np.isclose(policies.sum(), 1.0)
print(state.beliefs, policies)
```

`dispatch_belief_update` does not mutate the state it receives; the caller
builds the next `ModelState`. The variational update blends the prior and the
exact posterior with weight `learning_rate`, so with `learning_rate=0.5` the
beliefs move halfway toward the posterior at each observation. For the other agent classes and the helpers see
[[docs/api/api_examples|API examples]]; the full signatures are in
[[docs/api/api_reference|API reference]].
