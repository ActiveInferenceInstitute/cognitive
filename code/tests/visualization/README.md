---
title: Visualization tests
type: documentation
status: stable
updated: 2026-11-03
---

# `code/tests/visualization/`

[`test_continuous_generic.py`](test_continuous_generic.py) exercises
`Things.Continuous_Generic`: the agent and `ContinuousVisualizer` are driven and
every figure is written under a `tmp_path` fixture, never into the repository.
The figure tests carry `@pytest.mark.visualization`.

Run: `python -m pytest code/tests/visualization -q`. The suite-wide rules are in
[`../AGENTS.md`](../AGENTS.md).
