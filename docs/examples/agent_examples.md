---
title: Agent examples
type: guide
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[docs/api/agent_api]]"
      - "[[docs/api/api_examples]]"
      - "[[docs/examples/index]]"
---

# Agent examples

Three agent classes are importable. Each has an executed example in
[[docs/api/api_examples|API examples]].

| Agent | Import | Example |
|---|---|---|
| Discrete dispatcher over a generative model | `cognitive` | [[docs/examples/basic_api_usage|Basic API usage]] |
| `SimplePOMDP` | `Things.Simple_POMDP` | "Simple POMDP" in the API examples (run, save, load) |
| `ContinuousActiveInference` | `Things.Continuous_Generic` | "Continuous agent and animation" in the API examples |
| `HomeostaticInference` via `HomeostaticFactory` | `cognitive` | "Homeostatic agent" in the API examples |

Tests that exercise the same classes are in `code/tests/test_simple_pomdp.py`,
`code/tests/test_model_lifecycle.py`, and
`code/tests/visualization/test_continuous_generic.py`. The other folders under
`code/Things/` are documentation only; see
[[docs/examples/ant_colony_example|Ant colony usage example]] for the ant colony
status.
