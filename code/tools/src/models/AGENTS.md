---
title: Model module instructions
type: agents
status: stable
updated: 2026-11-03
---

# AGENTS.md — `code/tools/src/models/`

- `active_inference/` owns validated discrete inference and control;
  `matrices/` owns matrix operations, loading and initialization. Do not add a
  second implementation of either in another module.
- State vectors and matrices must be finite, non-negative and normalized before
  inference; reject invalid input instead of repairing it silently.
- Only discrete, mean-field and sampling inference are implemented. Do not
  document continuous or hierarchical dispatcher modes; continuous inference is
  `Things.Continuous_Generic`.
- A new public class needs a regression test and a documented import path.
- Parent rules: [../AGENTS.md](../AGENTS.md).
