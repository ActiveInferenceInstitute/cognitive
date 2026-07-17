---
title: Cognitive tools instructions
type: agents
status: stable
---

# Cognitive tools

The installable package source is `code/tools/src`, mapped to the package name
`cognitive` by the repository root `pyproject.toml`. Import public behavior
from `cognitive`, not from an internal filesystem name.

## Runtime boundaries

- `models/active_inference/` owns validated discrete inference and control.
- `models/matrices/` owns matrix operations, loading, and initialization.
- `utils/matrix_utils.py` owns shared probability helpers.
- `utils/create_node.py` owns configured knowledge-node creation.
- `utils/visualization/network_viz.py` owns deterministic knowledge graphs.
- `visualization/matrix_plots.py` owns explicit-output matrix figures.

Every new public behavior needs a real implementation, a regression test, a
documented import path, and configuration validation where configuration is
involved. Use temporary directories for generated artifacts. Run
`python -m pytest -q`, `ruff check .`, and `mypy code/tools/src code/Things
code/scripts` before submitting changes.
