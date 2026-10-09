---
title: Agent API reference
type: reference
status: stable
updated: 2026-10-09
---

# Agent API reference

Three agent classes are importable from the installed packages:

| Class | Import | Configuration | Step signature |
|---|---|---|---|
| `SimplePOMDP` | `Things.Simple_POMDP` | YAML path or dict with the sections listed in [[api_reference]] | `step(action=None) -> (observation, free_energy)` |
| `ContinuousActiveInference` | `Things.Continuous_Generic` | constructor arguments | `step(observation) -> (action, free_energy)` |
| `HomeostaticInference` | `cognitive` (via `HomeostaticFactory`) | YAML path or dict with `state_spaces`, `observation_model`, `transition_model`, `inference` | `step(action=None) -> (observation, free_energy)` |

`SimplePOMDP` and `HomeostaticInference` keep a `state` with `beliefs` and
persist it through `save_state` and `load_state`. `HomeostaticInference`
subclasses `ActiveInferenceModel`; `SimplePOMDP` and `ContinuousActiveInference`
are standalone classes.

Executed examples for each class are in [[api_examples]]. The other folders
under `code/Things/` (see [[code/Things/AGENTS|Things index]]) are conceptual
documentation without importable code.
