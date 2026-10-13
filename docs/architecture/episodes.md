---
title: Continuous and hierarchical episodes
type: reference
status: stable
updated: 2026-10-13
semantic_relations:
  - type: documents
    links:
      - "[[README|Architecture]]"
      - "[[../cli/reference|Command-line reference]]"
---

# Continuous and hierarchical episodes

Import these APIs from `cognitive.episodes`. They compose existing inference
engines into recorded perception-action loops; `PolicyType` remains discrete.

## Continuous

`run_continuous_episode(agent, steps, seed=0, initial_state=None,
process_dynamics=None, process_noise=0, sensor_noise=0, control_gain=1)` advances
the passed `Things.Continuous_Generic.ContinuousActiveInference` from its current
beliefs and time. It returns a `ContinuousTrajectory` with `time`, `states`,
`observations`, `actions`, `belief_means`, `free_energy` and reproduction `config`.
States, times and beliefs include the initial value; other arrays have one row per step.
`save_json` writes kind `continuous`; `save_csv` writes one row per step.

The world transition is

```text
x[t+1] = x[t] + dt * (process_dynamics @ x[t] - control_gain * action[t-1])
         + Normal(0, process_noise * sqrt(dt))
o[t] = agent.observation_matrix @ x[t+1] + Normal(0, sensor_noise)
```

The first previous action is zero, the initial state defaults to ones, and the
process dynamics default to negative identity. The agent's precision-weighted
residual action controls the next world step. This finite generalized-coordinate
approximation and Euler world are not an ODE solver or an expected-free-energy
action optimizer. Positive time steps are required but do not guarantee numerical
stability for arbitrary matrices or precisions; a non-finite run raises `ValueError`.

Steps and seeds must be integers in range; initial states and dynamics must be
finite with matching shapes; noise and control gain must be finite and non-negative.
Input rejection occurs before advancing the agent. A failure during integration
leaves the agent advanced through the failing step, so use a fresh agent for a retry.

## Hierarchical

`Hierarchy(lower, upper, goal_preferences, observation_map)` validates the interface
between two `DiscreteGenerativeModel` instances. Each upper action selects a row
of lower log preferences, with shape `(upper.num_actions, lower.num_observations)`.
The integer observation map assigns each lower observation to an upper observation.

`run_hierarchical_episode(hierarchy, lower_config, upper_config, steps,
decision_interval=5, seed=0)` returns a `HierarchicalTrajectory`. Its `lower` is a
standard `Trajectory`; `decision_steps`, `goals`, `upper_beliefs` and
`upper_posteriors` record the upper controller. Upper beliefs include the initial
value and the update after the final interval. `save_json` records both levels,
the upper model, its configuration, preferences and the cross-level mapping.

An upper policy selects a goal at the beginning of each interval. The lower
controller executes that preference for up to `decision_interval` world steps.
Its final state and posterior become the next interval's initial values. The
last lower observation updates the upper posterior after prediction under the
selected goal. No true world state is fed to the upper inference engine.
Upper sampling and lower intervals have separate child seeds. The supplied
models and configurations are not modified.

`corridor_hierarchy()` provides a two-region upper model controlling a four-cell
lower corridor. The upper transitions assume its selected endpoint can be
reached within an interval; this is an explicit planning approximation, not a
learned abstract dynamics model. Goal preferences change between intervals, so
the lower free-energy trace measures the goal active at each step. The lower
summary's preference rate refers to the original corridor's final endpoint.

Runnable examples: [07 and 08](../../examples/README.md).
