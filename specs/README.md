---
title: Specifications
type: documentation
status: stable
updated: 2026-11-03
---

# `specs/`

- [`discrete_variational.gnn.md`](discrete_variational.gnn.md) — the GNN
  declaration of the canonical discrete model.
- [`gnn-pin.json`](gnn-pin.json) — the upstream repository, tag (`v5.1.0`),
  commit `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`, and the SHA-256 digests of the
  parser and validator files used by conformance.

Validate with `python -m scripts.validate_gnn_spec --gnn-runtime <python>
--json`. The `gnn conformance (pinned)` CI job also checks that the tag
still resolves to the pinned commit before installing it. The model side of the
binding is in
[`../code/tools/src/models/active_inference/AGENTS.md`](../code/tools/src/models/active_inference/AGENTS.md).
