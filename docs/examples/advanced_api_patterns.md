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

The three inference methods share the discrete dispatcher.
`InferenceConfig.method` accepts `variational`, `sampling`, and `mean_field`;
`policy_aggregation` accepts `marginal` (default) or `min`.
The dispatcher supports only `policy_type="discrete"`. Continuous agents and
hierarchical episode composition are separate APIs, not dispatcher policy types.

All three policy methods use the model's action prior `E` unless `goal_prior`
replaces it for that call. The common target is proportional to
`(prior + 1e-12) * exp(-G / temperature)`, with the prior applied once.
Sampling approximates this target through sampled action counts.

`calculate_expected_free_energy(state)` deliberately returns prior-free
per-action energy `G`. Supplying its optional `goal_prior` returns
`G - temperature * log(prior + 1e-12)` instead.

Sampling belief updates draw Dirichlet samples only on positive posterior
support. Zero posterior components remain exactly zero; singleton support
returns the deterministic posterior.

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
    E=np.array([0.85, 0.15]),
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
