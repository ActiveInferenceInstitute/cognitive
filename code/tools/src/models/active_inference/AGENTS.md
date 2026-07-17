---
title: Discrete Active Inference runtime instructions
type: agents
status: stable
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
