---
title: Matrix plotting instructions
type: agents
status: stable
updated: 2026-11-03
---

# AGENTS.md — `code/tools/src/visualization/`

- `MatrixPlotter` requires an output directory and a style mapping. Do not add an
  implicit save path.
- Close every figure after writing it; tests run with the `Agg` backend from
  `code/tests/conftest.py`.
- Only `MatrixPlotter` is exported from `cognitive`. Import `StateSpacePlotter`
  and `NetworkPlotter` from `cognitive.visualization.matrix_plots`.
- A new plot needs a test that inspects the produced file or figure, marked
  `@pytest.mark.visualization`.
- Parent rules: [../AGENTS.md](../AGENTS.md).
