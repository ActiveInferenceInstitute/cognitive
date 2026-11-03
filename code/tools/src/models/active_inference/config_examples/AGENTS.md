---
title: Inference configuration instructions
type: agents
status: stable
updated: 2026-11-03
---

# AGENTS.md — `code/tools/src/models/active_inference/config_examples/`

- Keep `discrete_variational.yaml` loadable by the dispatcher's own loader. If a
  configuration field is renamed, update this file in the same change.
- Every matrix must satisfy the shape and normalization rules the model
  validates; a configuration that fails validation is not an example.
- Parent rules: [../AGENTS.md](../AGENTS.md).
