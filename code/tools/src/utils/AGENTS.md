---
title: Utility module instructions
type: agents
status: stable
updated: 2026-11-03
---

# AGENTS.md — `code/tools/src/utils/`

- `matrix_utils.py` owns the probability helpers. Do not copy one into another
  module; import it.
- Every helper states its normalization axis and rejects unknown constraints or
  invalid probability vectors.
- `create_node.py` renders knowledge nodes from the templates named in
  `config.yaml`; keep the console entry point `cognitive-create-node` working.
- `visualization/network_viz.py` must stay deterministic for a fixed input and
  write only to a caller-provided path.
- Parent rules: [../AGENTS.md](../AGENTS.md).
