---
title: Examples index
type: index
status: stable
updated: 2026-10-10
---

# Examples index

| Example | Source |
| --- | --- |
| Discrete generative model and dispatcher | [`README.md`](../../README.md) and [[docs/examples/basic_api_usage|Basic API usage]] |
| Inference methods, policy aggregation, goal prior | [[docs/examples/advanced_api_patterns|Advanced API patterns]] |
| Strict YAML dispatcher configuration | `code/tools/src/models/active_inference/config_examples/discrete_variational.yaml` and [[docs/examples/api_integration_examples|API integration examples]] |
| `SimplePOMDP`, `ContinuousActiveInference`, homeostatic agent | [[docs/api/api_examples|API examples]] |
| One-state POMDP and persistence | `code/tests/test_truth_audit.py` |
| Multi-frame visualization | `code/tests/test_truth_audit.py` |
| Continuous generalized-coordinate updates | `code/Things/Continuous_Generic/README.md` |
| Full figures, equations, citations, and PDF | [`docs/manuscript/README.md`](../manuscript/README.md) |

Run the examples after `python -m pip install -e ".[dev]"`. Each example uses
real constructors and the same validation rules as the runtime package.
