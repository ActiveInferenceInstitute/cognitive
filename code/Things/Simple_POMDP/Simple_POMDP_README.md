---
title: SimplePOMDP
type: reference
status: stable
updated: 2026-10-10
semantic_relations:
  - type: related_to
    links:
      - "[[docs/api/api_reference|API reference]]"
      - "[[code/Things/Simple_POMDP/AGENTS|Simple POMDP agent notes]]"
---

# SimplePOMDP

`SimplePOMDP` (`code/Things/Simple_POMDP/simple_pomdp.py`) is a seeded discrete
POMDP agent with exact Bayesian belief updates and one-step expected free
energy action selection. It is importable as `Things.Simple_POMDP`, which
exports `SimplePOMDP`, `SimplePOMDPState`, and `compute_expected_free_energy`.

## Files

```text
code/Things/Simple_POMDP/
  __init__.py            exports
  simple_pomdp.py        agent, state dataclass, plotter
  README.md              short overview
  Simple_POMDP_README.md this file
  AGENTS.md              agent notes
```

There is no bundled configuration file; configurations are a YAML path or a
dict. Tests live in `code/tests/test_simple_pomdp.py`.

## Configuration

The configuration must contain exactly the sections `model`, `state_space`,
`observation_space`, `action_space`, `matrices`, `inference`, and
`visualization`, plus an optional root `seed`. Unknown sections or fields raise
`ValueError`.

| Section | Accepted fields |
|---|---|
| `model` | `name`, `description`, `version` |
| `state_space` | `num_states`, `state_labels`, `initial_state` |
| `observation_space` | `num_observations`, `observation_labels` |
| `action_space` | `num_actions`, `action_labels` |
| `inference` | `learning_rate`, `temperature`, `seed`, `time_horizon` / `temporal_horizon` (must be 1) |
| `visualization` | `output_dir`, `formats`, `dpi`, `style` |

Every `inference` field is either used or rejected with a `ValueError`:

| Field | Effect |
|---|---|
| `learning_rate` | Weight of the exact Bayes posterior in the belief update, in `[0, 1]`. |
| `temperature` | Positive number. `step()` samples the action from `softmax(-(G - ln E) / temperature)`, where `G` is the one-step expected free energy and `E` the action prior. Omit it for greedy selection, the zero-temperature limit. |
| `seed` | Seeds the generator behind transitions, observations, and sampled actions. The root-level `seed` is equivalent; giving both with different values is an error. |
| `time_horizon`, `temporal_horizon` | Must be 1 when present: the agent scores single actions. Use `ActiveInferenceDispatcher` for multi-step policies. |
| `num_iterations`, `policy_learning_rate` | Rejected: the belief update is one exact Bayes step and `E` is never learned, so nothing would read them. |

`matrices` must define exactly `A_matrix` to `E_matrix`;
each takes `shape`, `initialization` (`uniform`, `random`,
`random_stochastic`, `identity_based`, `log_preferences`),
`initialization_params`, `constraints` (`non_negative`, `column_stochastic`,
`sum_to_one`; `row_stochastic` for 2-D matrices only), `description`, and
`learning_rate`.

## Matrix conventions

- `A[o, s]`: observation likelihood, shape `(observations, states)`, column
  stochastic.
- `B[s_next, s_prev, a]`: transitions, shape `(states, states, actions)`,
  column stochastic for every action.
- `C`: one finite log-preference per observation.
- `D`: prior over states, normalised on construction.
- `E`: prior over actions, normalised on construction.

## Behaviour

- `step(action=None)` samples the next state from `B`, samples an observation
  from `A`, applies a Bayes update of the beliefs, and returns
  `(observation, free_energy)`. With no action, it picks one by one-step
  expected free energy minus the log action prior: greedy when `temperature` is
  absent, softmax-sampled when it is set.
- `learning_rate` blends the Bayesian posterior into the previous belief.
- `run(steps, actions=None)` calls `step` repeatedly; `reset()` restores the
  prior, time, and history.
- `save_state(path)` and `load_state(path)` write and read versioned YAML
  (`schema_version: 1`) including the history.
- The seed defaults to 7 when neither the root nor `inference` sets one.
- `state.history` records `states`, `observations`, `actions`, `beliefs`,
  `free_energy`, `efe_total`, `efe_epistemic`, and `efe_pragmatic`.

## Example

```python
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from Things.Simple_POMDP import SimplePOMDP

with tempfile.TemporaryDirectory() as tmp:
    config = {
        "model": {"name": "two_state", "version": "1"},
        "state_space": {"num_states": 2, "state_labels": ["a", "b"], "initial_state": 0},
        "observation_space": {"num_observations": 2, "observation_labels": ["a", "b"]},
        "action_space": {"num_actions": 2, "action_labels": ["stay", "switch"]},
        "matrices": {
            "A_matrix": {"shape": [2, 2], "initialization": "identity_based"},
            "B_matrix": {"shape": [2, 2, 2], "initialization": "identity_based"},
            "C_matrix": {
                "shape": [2],
                "initialization": "log_preferences",
                "initialization_params": {"preferences": [0.0, 1.0]},
            },
            "D_matrix": {"shape": [2], "initialization": "uniform"},
            "E_matrix": {"shape": [2], "initialization": "uniform"},
        },
        "inference": {"learning_rate": 1.0, "temperature": 1.0},
        "visualization": {"output_dir": str(Path(tmp) / "plots")},
        "seed": 4,
    }
    agent = SimplePOMDP(config)
    observation, free_energy = agent.step()
    agent.run(4)
    assert agent.state.time_step == 5

    agent.visualize("belief_evolution")
    assert (Path(tmp) / "plots" / "belief_evolution.png").exists()
```

## Visualisation

`visualize(plot_type)` saves `<output_dir>/<plot_type>.png` and returns the
closed figure. Supported plot types:

- `belief_evolution`
- `free_energy_landscape`
- `policy_evaluation`
- `state_transitions`
- `observation_likelihood`
- `efe_components_detailed`

Any other name raises `ValueError`.

## Tests

```bash
.venv/bin/python -m pytest code/tests/test_simple_pomdp.py -q
```

Dependencies are NumPy, Matplotlib, and PyYAML, plus the `cognitive` package
for `DiscreteGenerativeModel` and matrix helpers.
