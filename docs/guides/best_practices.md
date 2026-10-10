---
title: Best practices guide
type: guide
status: stable
updated: 2026-10-10
tags:
  - best_practices
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[docs/guides/api_implementation]]"
      - "[[docs/guides/agent_development]]"
      - "[[docs/implementation/README]]"
---

# Best practices

Practices that the `cognitive` package enforces or that its tests check. Each
rule names the behaviour it relies on.

## Build the generative model from validated arrays

- `A[o, s]` is `P(o | s)` and every column sums to one.
- `B[s_next, s_prev, a]` has one column-stochastic slice per action.
- `C` holds log-preferences over observations; `D` and `E` are state and
  action priors.
- `DiscreteGenerativeModel` checks these conditions on construction and raises
  `ValueError` for wrong shapes, negative or non-finite entries, and columns
  that do not sum to one. Let the constructor reject bad input rather than
  renormalizing it silently.

```python
import numpy as np

from cognitive import DiscreteGenerativeModel

model = DiscreteGenerativeModel(
    A=np.array([[0.9, 0.1], [0.1, 0.9]]),
    B=np.stack([np.eye(2)], axis=2),
    C=np.array([0.0, 1.0]),
    D=np.array([0.5, 0.5]),
    E=np.array([1.0]),
)
try:
    model.posterior(0.5, model.D)
except ValueError as error:
    print(error)  # Observation must be an integer, got 0.5
```

## Pass indices as integers

Observation and action indices are not truncated: `1.9` raises, while `1`,
`np.int64(1)` and `1.0` are accepted. See the Unreleased behaviour changes in
[`CHANGELOG.md`](../../CHANGELOG.md).

## Configure inference explicitly

- Set `method` (`variational`, `sampling`, or `mean_field`), `temporal_horizon`,
  `learning_rate`, and `precision_init` in `InferenceConfig`; out-of-range values
  raise `ValueError`.
- `exploration_weight` defaults to `0.0`, the canonical objective. Raise it only
  when extra information seeking is intended.
- `temporal_horizon` enumerates every action sequence, so the number of
  policies grows as `num_actions ** horizon`. `policy_limit` (default 4096)
  caps enumeration, and a limit too small to cover every first action raises
  instead of zeroing an action.
- Pass `seed` for reproducible sampling; the dispatcher builds its generator
  from `InferenceConfig.seed` unless an `rng` is supplied.

## Test the behaviour you rely on

- The suite lives in `code/tests/`; run `python -m pytest -q`. Coverage uses the
  configured floor: `python -m pytest -q --cov` (do not pass `--cov=<path>`,
  which narrows what is measured).
- Assert that outputs are finite and normalized, as in
  `code/tests/test_truth_audit.py`.
- Test the smallest valid model and the invalid inputs that must raise.

## Document with checks

Run `python code/scripts/validate_docs.py --json` and
`python code/scripts/verify_links.py . --json` before merging documentation
changes. See [[docs/repo_docs/documentation_standards|Documentation standards]].

## Related

- [[docs/guides/agent_development|Agent development guide]]
- [[docs/guides/api_implementation|API implementation guide]]
- [[docs/implementation/README|Implementation guides]]
