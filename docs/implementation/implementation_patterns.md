---
title: Implementation patterns
type: guide
status: stable
created: 2024-02-06
updated: 2026-10-10
tags:
  - implementation
  - patterns
semantic_relations:
  - type: relates
    links:
      - "[[docs/api/api_reference|API reference]]"
      - "[[docs/guides/api_implementation|API implementation guide]]"
      - "[[docs/guides/agent_development|Agent development guide]]"
      - "[[knowledge_base/cognitive/active_inference|Active inference]]"
---

# Implementation patterns

Patterns that the package itself follows, each checked by the code snippet or the
test named next to it. The class and function names are listed in
[[docs/api/api_reference|the API reference]].

## Validate at construction, fail loudly

`DiscreteGenerativeModel` and `InferenceConfig` reject inconsistent shapes and
out-of-range values with `ValueError` instead of repairing them. Unknown keys in a
YAML configuration are rejected too. The following runs:

```python
import numpy as np

from cognitive import InferenceConfig

try:
    InferenceConfig("variational", "hierarchical", 2, 0.5, 1.0)
except ValueError as error:
    print("rejected:", error)
else:
    raise AssertionError("only the discrete policy type is supported")
```

## Seed every stochastic component

`InferenceConfig(seed=...)` (default `0`) seeds the dispatcher's random generator,
and `SimplePOMDP` and `ContinuousActiveInference` accept a `seed`. Two
dispatchers built with the same seed produce the same sampling-method output:

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
outputs = []
for _ in range(2):
    dispatcher = ActiveInferenceDispatcher(
        InferenceConfig("sampling", "discrete", 2, 0.5, 1.0, seed=3), model
    )
    state = ModelState(model.D.copy(), model.E.copy(), 1.0, 0.0, 0.0)
    outputs.append(dispatcher.dispatch_belief_update(1, state))
assert np.allclose(outputs[0], outputs[1])
```

## Check that probability outputs are normalized

Beliefs and policy distributions returned by the dispatcher sum to one; assert
that in any new code path rather than trusting it. See the dispatcher example in
[[docs/api/api_reference|the API reference]].

## Persist and restore state explicitly

`save_state` and `load_state` on `SimplePOMDP` and `ActiveInferenceModel`
round-trip a validated state through YAML; the round trip is tested in
`code/tests/test_model_lifecycle.py`. See
[[docs/api/api_examples|the API examples]] for a worked case.

## Test against an independent calculation

New maths gets a test that compares it with a closed form or an algebraically
equivalent computation, not with its own output
(`code/tests/test_math_oracles.py`). The mutation check
(`cognitive-mutation-check`) records each defect that was fixed so the test that
caught it stays load-bearing.

## Related

- [[docs/implementation/README|Implementation index]]
- [[docs/repo_docs/unit_testing|Unit testing]]
- [[docs/repo_docs/naming_conventions|Naming conventions]]
