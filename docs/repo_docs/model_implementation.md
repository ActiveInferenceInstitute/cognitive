---
title: Model implementation
type: guide
status: stable
created: 2024-02-06
updated: 2026-10-10
tags:
  - implementation
  - models
  - development
semantic_relations:
  - type: implements
    links:
      - "[[knowledge_base/cognitive/active_inference|Active Inference]]"
      - "[[knowledge_base/mathematics/free_energy_principle|Free Energy Principle]]"
  - type: relates
    links:
      - "[[docs/implementation/implementation_patterns|Implementation patterns]]"
      - "[[docs/api/api_reference|API reference]]"
---

# Model implementation

How a model is built on this package. The theory is in the knowledge base
([[knowledge_base/cognitive/active_inference|Active Inference]]); the
signatures are in [[docs/api/api_reference|the API reference]]. This page covers
the extension points.

## The pieces

| Piece | Where | Role |
|---|---|---|
| `DiscreteGenerativeModel` | `code/tools/src/models/active_inference/generative_model.py` | Validated `A`, `B`, `C`, `D`, `E` arrays; prediction, posterior, expected free energy, policy evaluation |
| `InferenceConfig` | `dispatcher.py` | Method (`variational`, `mean_field`, `sampling`), horizon, learning rate, precision, seed |
| `ActiveInferenceDispatcher` | `dispatcher.py` | `dispatch_belief_update` and `dispatch_policy_inference` for a model and config |
| `ModelState` | `base.py` | Beliefs, policies, precision, free energy, prediction error |
| `ActiveInferenceModel` | `base.py` | Abstract lifecycle: belief and policy updates, precision, free energy, state save and load |

Matrix conventions: `A[o, s] = P(o | s)`, `B[s_next, s_prev, a]`, `C` holds
log-preferences over observations, `D` and `E` are state and action priors.
Only discrete policies are supported.

## Writing a model

Subclass `ActiveInferenceModel`. Five methods are abstract: `_load_config`,
`_initialize_matrices`, `_initialize_state`, `step`, and `visualize`. The base
class supplies `update_beliefs`, `infer_policies`, `update_precision`,
`calculate_free_energy`, `get_state`, `save_state`, and `load_state` (a
versioned YAML document), provided the subclass sets `self.dispatcher`.

```python
import numpy as np

from cognitive import (
    ActiveInferenceDispatcher,
    ActiveInferenceModel,
    DiscreteGenerativeModel,
    InferenceConfig,
    ModelState,
)


class TwoStateModel(ActiveInferenceModel):
    """A minimal concrete model that routes inference through the dispatcher."""

    def _load_config(self, config_path):
        return dict(config_path)

    def _initialize_matrices(self):
        self.generative_model = DiscreteGenerativeModel(
            A=np.array([[0.9, 0.1], [0.1, 0.9]]),
            B=np.stack([np.eye(2), np.array([[0.1, 0.9], [0.9, 0.1]])], axis=2),
            C=np.array([0.0, 1.0]),
            D=np.array([0.5, 0.5]),
            E=np.array([0.5, 0.5]),
        )
        config = InferenceConfig("variational", "discrete", 2, 0.5, 1.0, seed=0)
        self.dispatcher = ActiveInferenceDispatcher(config, self.generative_model)

    def _initialize_state(self):
        model = self.generative_model
        return ModelState(model.D.copy(), model.E.copy(), 1.0, 0.0, 0.0)

    def step(self, action=None):
        observation = 1
        self.update_beliefs(observation)
        return observation, float(self.state.free_energy)

    def visualize(self, plot_type, **kwargs):
        return self.state.beliefs


model = TwoStateModel({"prior_beliefs": [0.5, 0.5]})
observation, free_energy = model.step()
policies = model.infer_policies()
assert np.isclose(model.state.beliefs.sum(), 1.0) and np.isclose(policies.sum(), 1.0)
```

The constructor calls `_load_config`, then `_initialize_matrices`, then
`_initialize_state`, and validates the state: beliefs and policies must be
finite, non-negative, normalized vectors, `precision` must be positive, and
`prediction_error` non-negative. A violation raises `ValueError`.

## Complete implementations to read

- `Things.Simple_POMDP.SimplePOMDP` (`code/Things/Simple_POMDP/simple_pomdp.py`):
  a YAML- or dict-configured discrete agent with `step`, `run`, `reset`,
  `save_state`, `load_state`, and `visualize`.
- `Things.Continuous_Generic.ContinuousActiveInference`
  (`code/Things/Continuous_Generic/continuous_generic.py`): a
  generalized-coordinates continuous agent.
- `HomeostaticFactory` in `homeostatic.py`: builds a homeostatic agent from a
  labelled configuration.

The other folders under `code/Things/` document agent designs but contain no
importable code.

## Checklist for a change

1. Write the mathematical definition in the docstring or the knowledge-base page.
2. Validate inputs at the boundary and raise `ValueError` naming the parameter.
3. Add a deterministic regression test with a closed-form or hand-computed
   oracle ([[docs/repo_docs/unit_testing|Unit testing]]).
4. Update [[docs/api/api_reference|the API reference]] and `CHANGELOG.md`.

## Related documentation

- [[docs/api/api_reference|API reference]]
- [[docs/implementation/implementation_patterns|Implementation patterns]]
- [[docs/guides/agent_development|Agent development guide]]
- [[docs/repo_docs/unit_testing|Unit testing]]
