---
title: Advanced API patterns
type: guide
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[docs/api/api_reference]]"
      - "[[docs/examples/basic_api_usage]]"
---

# Advanced API patterns

The three inference methods, policy aggregation, and a goal prior all go through
the same dispatcher. `InferenceConfig` accepts `variational`, `sampling`, and
`mean_field`, and `policy_aggregation` accepts `marginal` (default) or `min`.
Only discrete policies are implemented; hierarchical or user-defined inference
methods are not part of the package.

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
state = ModelState(np.array([0.9, 0.1]), model.E.copy(), 1.0, 0.0, 0.0)

for method in ("variational", "mean_field", "sampling"):
    for aggregation in ("min", "marginal"):
        config = InferenceConfig(
            method, "discrete", 2, 0.5, 1.0,
            policy_aggregation=aggregation, seed=7,
        )
        dispatcher = ActiveInferenceDispatcher(config, model)
        policies = dispatcher.dispatch_policy_inference(state)
        assert np.isclose(policies.sum(), 1.0)
        print(method, aggregation, policies.round(3))

# goal_prior is a distribution over actions that replaces the action prior E
# for one policy-inference call.
goal = np.array([0.9, 0.1])
dispatcher = ActiveInferenceDispatcher(
    InferenceConfig("variational", "discrete", 2, 0.5, 1.0, seed=7), model
)
print(dispatcher.dispatch_policy_inference(state, goal_prior=goal).round(3))
```

Supporting pieces:

- `ActiveInferenceFactory.create_from_yaml` builds a dispatcher from a strict YAML
  mapping; a working file is
  `code/tools/src/models/active_inference/config_examples/discrete_variational.yaml`.
- `InferenceConfig(temporal_horizon=n)` plans over every action sequence of
  length `n`, capped by `policy_limit`.
- `HomeostaticFactory` and `ContinuousActiveInference` are shown in
  [[docs/api/api_examples|API examples]].
