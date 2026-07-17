# Dispatcher reference

`ActiveInferenceDispatcher` requires both an `InferenceConfig` and a
validated `DiscreteGenerativeModel`:

```python
import numpy as np

from cognitive.models.active_inference import (
    ActiveInferenceDispatcher,
    DiscreteGenerativeModel,
    InferenceConfig,
)

model = DiscreteGenerativeModel(
    A=np.eye(2),
    B=np.stack([np.eye(2)], axis=2),
    C=np.zeros(2),
    D=np.array([0.5, 0.5]),
    E=np.array([1.0]),
)
dispatcher = ActiveInferenceDispatcher(
    InferenceConfig("mean_field", "discrete", 2, 0.5, 1.0, seed=3), model
)
```

The supported methods are `variational`, `mean_field`, and `sampling`.
Policy inference evaluates action sequences up to the configured horizon and
returns a normalized distribution over their first actions. Risk, ambiguity,
and epistemic information gain are calculated from the generative model.
Continuous models use the dedicated `Things.Continuous_Generic`
implementation instead of this dispatcher.
