---
title: Agent development guide
type: guide
status: stable
---

# Agent development guide

## Choose a maintained runtime

- Use `cognitive.DiscreteGenerativeModel` and
  `cognitive.ActiveInferenceDispatcher` for finite categorical inference.
- Use `Things.Simple_POMDP.SimplePOMDP` when a self-contained discrete agent
  with histories and visualization is needed.
- Use `Things.Continuous_Generic.ContinuousActiveInference` for continuous
  generalized-coordinate updates.
- Use `HomeostaticFactory` for labelled spaces and control priors.

## Development sequence

1. Specify matrix shapes and stochastic orientation.
2. Load the configuration with strict schema validation.
3. Run the inference path and assert finite normalized outputs.
4. Persist and reload state when a run must be resumed.
5. Add regression tests for invalid input and the smallest valid model.
6. Document the public import and add a runnable example.

The complete mathematical correspondence and executable evidence are in
[`docs/manuscript/`](../manuscript/README.md). The root package README contains
the smallest working discrete example.
