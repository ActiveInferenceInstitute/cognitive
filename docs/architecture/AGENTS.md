---
title: Architecture documentation instructions
description: Rules for editing the architecture documentation so that it describes the code that exists.
type: agents
status: stable
updated: 2026-10-13
semantic_relations:
  - type: documents
    links:
      - "[[README|Architecture]]"
---

# Architecture documentation instructions

## Ground truth

The module table and the data-flow diagram in [`README.md`](README.md)
describe `code/tools/src/settings.py`, `simulation.py`, `cli.py` and
`visualization/trajectory_plots.py`. When one of them changes its inputs or
outputs, change the diagram in the same commit.

## Rules

1. Describe what the code does today. Do not describe planned modules.
2. Keep the per-step order in "One step of an episode" equal to the order in
   the docstring of `simulation.py`.
3. A claim about reproducibility must be backed by a test; see
   `code/tests/test_simulation.py` and `code/tests/test_settings.py`.
4. Link out to the settings and CLI references instead of repeating them.
