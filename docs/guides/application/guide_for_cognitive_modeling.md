---
title: Guide for cognitive modelling
type: guide
status: stable
---

# Guide for cognitive modelling

## From theory to executable model

Start by defining the hidden state, observation, and action sets. For a finite
model, encode them in `A`, `B`, `C`, `D`, and `E` using the conventions in
`docs/manuscript/01_introduction.md`. Validate the model before running
inference; the constructor rejects incompatible shapes and invalid mass.

```python
import numpy as np

from cognitive import DiscreteGenerativeModel

model = DiscreteGenerativeModel(
    A=np.eye(2),
    B=np.stack([np.eye(2)], axis=2),
    C=np.array([0.0, 1.0]),
    D=np.array([0.5, 0.5]),
    E=np.array([1.0]),
)
posterior = model.posterior(0, model.D)
assert np.isclose(posterior.sum(), 1.0)
```

Use `ActiveInferenceDispatcher` for belief and policy inference, or
`HomeostaticFactory` when the state space has labels and control bounds. Use
`ContinuousActiveInference` when the latent variables are continuous and a
generalized-coordinate update is appropriate.

## Reporting

Record the configuration, seed, matrix shapes, inference method, policy
horizon, and output hashes. The executable manuscript under
`docs/manuscript/` provides a complete publication pattern with formal
equations, bibliography, figures, and validation evidence.
