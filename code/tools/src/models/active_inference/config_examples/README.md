---
title: Inference configuration examples
type: documentation
status: stable
updated: 2026-11-03
---

# `code/tools/src/models/active_inference/config_examples/`

[`discrete_variational.yaml`](discrete_variational.yaml) is the canonical
discrete variational configuration, bound to the GNN declaration in
[`../../../../../../specs/discrete_variational.gnn.md`](../../../../../../specs/discrete_variational.gnn.md)
by `code/tests/test_gnn_conformance.py`. It holds inference settings (`method`, `policy_type`,
`temporal_horizon`, `learning_rate`, `precision_init`, `temperature`,
`discount_factor`, `exploration_weight`, `policy_limit`, `seed`) and a
`generative_model` block holding the `A`, `B`, `C`, `D` and `E` matrices.

The dispatcher that consumes it is documented in
[`../docs/README.md`](../docs/README.md); the model classes are in
[`../README.md`](../README.md).
