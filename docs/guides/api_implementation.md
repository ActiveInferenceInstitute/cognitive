---
title: API implementation guide
type: guide
status: stable
---

# API implementation guide

The supported implementation pattern starts with a validated generative
model, passes it to a configured dispatcher, and checks the resulting
probability vectors.

```python
import numpy as np

from cognitive import ActiveInferenceDispatcher, DiscreteGenerativeModel, InferenceConfig, ModelState

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
posterior = dispatcher.dispatch_belief_update(1, state)
policy = dispatcher.dispatch_policy_inference(state)
assert np.isclose(posterior.sum(), 1.0)
assert np.isclose(policy.sum(), 1.0)
```

Use `HomeostaticFactory` for labelled state spaces and
`Things.Continuous_Generic.ContinuousActiveInference` for continuous
generalized-coordinate dynamics. Configuration files are strict: unknown
keys are errors, relative paths resolve from the configuration file, and
random operations accept an explicit seed.

For a complete end-to-end example, run
`cognitive-build-manuscript --output build/manuscript`. For profiling, run
`cognitive-benchmark --repetitions 10` and inspect its JSON output.
