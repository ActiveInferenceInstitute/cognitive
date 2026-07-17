---
title: Cognitive model modules
type: documentation
status: stable
---

# Cognitive model modules

The installable model modules are exposed through the `cognitive` package.
The directory contains the validated discrete Active Inference runtime and
matrix operations used by it.

## Public modules

- `cognitive.models.active_inference.generative_model.DiscreteGenerativeModel`
  validates `A`, `B`, `C`, `D`, and `E` and evaluates posteriors and policies.
- `cognitive.models.active_inference.dispatcher.ActiveInferenceDispatcher`
  implements variational, mean-field, and sampling inference for discrete
  policies.
- `cognitive.models.active_inference.homeostatic.HomeostaticInference`
  adds labelled state spaces and homeostatic or adaptive control.
- `cognitive.models.matrices.matrix_ops.MatrixOps` provides matrix
  normalization, entropy, KL divergence, softmax, loading, and initialization.

The public symbols and their import paths are defined in
`code/tools/src/models/active_inference/__init__.py` and
`code/tools/src/models/matrices/__init__.py`. Runtime behavior is protected by
the tests in `code/tests/` and the examples in the repository root
`README.md`.

## Matrix conventions

`A[o, s]` is `P(o | s)`, `B[s_next, s_prev, action]` is
`P(s_next | s_prev, action)`, `C` contains observation log-preferences, `D`
is a normalized state prior, and `E` is a normalized action prior. Callers
must supply finite arrays; invalid shapes, constraints, or distributions raise
an exception with a diagnostic message.
