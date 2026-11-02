---
title: Discrete Active Inference runtime
type: documentation
status: stable
updated: 2026-10-10
---

# Discrete Active Inference runtime

The active-inference package contains four cooperating pieces:

1. `DiscreteGenerativeModel` validates `A`, `B`, `C`, `D`, and `E` and
   implements prediction, exact Bayesian posteriors, information metrics, and
   finite-horizon policy evaluation.
2. `InferenceConfig` validates method, horizon, learning, precision,
   temperature, sampling, and seed settings.
3. `ActiveInferenceDispatcher` implements variational, mean-field, and
   particle-sampling belief updates plus normalized first-action policy
   distributions.
4. `ActiveInferenceModel` and `HomeostaticInference` provide validated state
   lifecycle, persistence, target handling, and action selection.

Minimal construction:

```python
import numpy as np

from cognitive.models.active_inference import (
    ActiveInferenceDispatcher,
    DiscreteGenerativeModel,
    InferenceConfig,
    ModelState,
)

model = DiscreteGenerativeModel(
    A=np.eye(2),
    B=np.stack([np.eye(2)], axis=2),
    C=np.zeros(2),
    D=np.array([0.5, 0.5]),
    E=np.array([1.0]),
)
dispatcher = ActiveInferenceDispatcher(
    InferenceConfig("variational", "discrete", 1, 1.0, 1.0), model
)
state = ModelState(model.D.copy(), model.E.copy(), 1.0, 0.0, 0.0)
posterior = dispatcher.dispatch_belief_update(0, state)
assert np.isclose(posterior.sum(), 1.0)
```

Only the validated discrete dispatcher modes documented above are exported.

## Model declaration

The canonical discrete model is also declared in Generalized Notation Notation in
[`specs/discrete_variational.gnn.md`](../../../../../specs/discrete_variational.gnn.md).
That declaration is the canonical statement of `A`, `B`, `C`, `D`, and `E`; the
configuration in
[`config_examples/discrete_variational.yaml`](config_examples/discrete_variational.yaml)
stays the runtime configuration, and
[`code/tests/test_gnn_conformance.py`](../../../../tests/test_gnn_conformance.py)
fails when the two disagree. `gnn_spec.py` reads the declaration with the
standard library plus NumPy; the format's parser revision is pinned in
[`specs/gnn-pin.json`](../../../../../specs/gnn-pin.json). See
[`docs/gnn.md`](../../../../../docs/gnn.md) for the commands.
