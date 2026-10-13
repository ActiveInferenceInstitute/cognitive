---
title: Example configuration instructions
description: Rules for editing the example settings and model files.
type: agents
status: stable
updated: 2026-10-13
semantic_relations:
  - type: documents
    links:
      - "[[README|Example configurations]]"
---

# Example configuration instructions

- Every key in `settings.yaml` must be a real key of `cognitive.settings`;
  unknown keys are errors. `code/tests/test_examples.py` runs
  `01_run_episode.py`, which loads both files.
- `two_state.yaml` must load with `cognitive.simulation.load_model`, which
  rejects unknown keys and a file that defines both or neither of `B` and
  `B_by_action`.
- When a file is added here, list it in [`../README.md`](../README.md).
