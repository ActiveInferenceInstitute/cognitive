---
title: Discrete Active Inference runtime instructions
type: agents
status: stable
updated: 2026-10-10
---

# Discrete Active Inference runtime

`DiscreteGenerativeModel` owns validated `A`, `B`, `C`, `D`, and `E` arrays.
`ActiveInferenceDispatcher` owns the three implemented discrete inference
methods and finite-horizon policy evaluation. `HomeostaticInference` adapts
the same contracts to labelled state spaces.

Every public state vector must be finite, non-negative, and normalized before
it is used for inference. New inference behavior requires a mathematical
definition, a deterministic regression case where applicable, and a public
documentation example.

## Model declaration

`gnn_spec.py` reads the committed GNN declaration in
`specs/discrete_variational.gnn.md` with the standard library plus NumPy and no
upstream import; `specs/gnn-pin.json` pins the upstream parser revision the
declaration is checked against.
`code/tests/test_gnn_conformance.py` binds the declaration to
`config_examples/discrete_variational.yaml` through
`ActiveInferenceFactory.create_from_yaml`, so editing one side alone fails the
suite, and `code/scripts/validate_gnn_spec.py` runs the pinned parser whenever a
runtime for it is named. Change the declaration and the canonical configuration
together, then follow `docs/gnn.md`.
