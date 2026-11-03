---
title: Specification instructions
type: agents
status: stable
updated: 2026-11-03
---

# AGENTS.md — `specs/`

- The declaration and
  `code/tools/src/models/active_inference/config_examples/discrete_variational.yaml`
  are one contract, bound by `code/tests/test_gnn_conformance.py`. Change them
  together.
- A re-pin updates the tag, the commit and every parser hash in `gnn-pin.json`,
  and must re-run `python -m scripts.validate_gnn_spec --gnn-runtime <python>`.
- Validate against the pinned revision, never a live GNN working tree.
- Parent rules: [../AGENTS.md](../AGENTS.md).
