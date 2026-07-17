---
title: Cognitive API reference
type: reference
status: stable
---

# Cognitive API reference

The package name is `cognitive`. The following symbols are exported from
`cognitive.__init__` and are covered by the repository test suite.

## Discrete model and dispatcher

```python
import numpy as np

from cognitive import ActiveInferenceDispatcher, DiscreteGenerativeModel, InferenceConfig

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
```

`DiscreteGenerativeModel` provides `posterior`, `predict_states`,
`predict_observations`, `expected_free_energy`, `enumerate_policies`, and
`evaluate_policy`. `ActiveInferenceDispatcher` provides
`dispatch_belief_update`, `dispatch_policy_inference`, and
`update_precision`. `InferenceConfig` rejects unsupported method or policy
values, invalid horizons, temperatures, precisions, sample counts, and
unknown configuration keys when loaded from YAML by
`ActiveInferenceFactory.create_from_yaml`.

## Agent families

- `Things.Simple_POMDP.SimplePOMDP` loads a strict configuration, initializes
  validated matrices, runs seeded discrete steps, records histories, and
  persists state.
- `Things.Continuous_Generic.ContinuousActiveInference` performs
  precision-weighted generalized-coordinate updates with configurable
  dynamics, observation mapping, precision, timestep, and seed.
- `Things.Continuous_Generic.ContinuousVisualizer` writes figures and
  multi-frame Pillow animations from validated histories.

## Utilities and controls

`HomeostaticFactory.create_basic` and `create_adaptive` construct validated
`HomeostaticInference` agents. `MatrixOps`, `MatrixLoader`, and
`MatrixInitializer` are available from `cognitive.models.matrices.matrix_ops`.
Probability helpers are available from `cognitive.utils.matrix_utils`.

## Commands

```bash
cognitive-create-node --help
cognitive-verify-links . --json
cognitive-validate-docs . --json
cognitive-benchmark --repetitions 10
cognitive-build-manuscript --output build/manuscript
```

These commands are declared in `pyproject.toml`; no separate dependency file
or alternate import package is required.
