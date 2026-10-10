---
title: Continuous Generic active inference
type: reference
status: stable
updated: 2026-10-10
semantic_relations:
  - type: related_to
    links:
      - "[[docs/api/api_reference|API reference]]"
---

# Continuous Generic active inference

`ContinuousActiveInference` (`code/Things/Continuous_Generic/continuous_generic.py`)
is a NumPy-only agent that performs precision-weighted gradient descent on a
free energy defined over generalized coordinates. It contains no neural
networks, no learned generative or recognition model, and no gradient-based
action optimiser. `ContinuousVisualizer` (`visualization.py`) renders its
histories. Both are exported from `Things.Continuous_Generic`, together with
`ContinuousState`.

## Files

```text
code/Things/Continuous_Generic/
  __init__.py                exports
  continuous_generic.py      agent and ContinuousState
  visualization.py           ContinuousVisualizer
  README.md                  short overview
  Continuous_Generic_README.md  this file
  AGENTS.md                  agent notes
```

There is no configuration file and no `requirements.txt`; the agent is
configured by constructor arguments. A visualisation test lives in
`code/tests/visualization/test_continuous_generic.py`.

## Model

Beliefs are a matrix `belief_means` of shape `(n_states, n_orders)`. Column 0
is the latent state and later columns are its successive derivatives. The free
energy computed by the agent is

```text
F = 1/2 * sum_i Pi_obs[i] * (o - G mu_0)[i]^2
    + dynamics_weight * 1/2 * sum_k sum_i Pi_state[i, k] * (mu_{k+1} - D mu_k)[i]^2
```

with `G` the observation matrix, `D` the dynamics matrix, and `Pi` the
precisions. `step(observation)` moves the beliefs by `dt * alpha * (-dF/dmu)`,
optionally adapts the observation precision, advances `state.time` by `dt`, and
returns `(action, free_energy)`. The returned action is the precision-weighted
prediction error projected through `G.T`, an array of shape `(n_states,)`.

## Constructor arguments

| Argument | Default | Meaning |
|---|---|---|
| `n_states`, `n_obs`, `n_orders` | 2, 2, 3 | positive integers |
| `dt` | 0.01 | step length, positive |
| `alpha` | 0.1 | gradient step size, positive |
| `observation_matrix` | identity-like | shape `(n_obs, n_states)` |
| `dynamics_matrix` | identity | shape `(n_states, n_states)` |
| `observation_precision`, `state_precision` | 1.0 | positive scalar or vector |
| `precision_learning_rate` | 0.0 | in `[0, 1]`; positive values adapt the observation precision |
| `dynamics_weight` | 1.0 | non-negative weight of the dynamic term |
| `seed` | 0 | non-negative integer or `None` |
| `precision_min`, `precision_max` | 1e-3, 1e3 | clip range for learned precision |

Invalid values raise `ValueError`. `reset(belief_means=None)` restores beliefs
and time. `state` is a `ContinuousState` with `belief_means`,
`belief_precisions`, and `time`.

## Example

```python
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from Things.Continuous_Generic import ContinuousActiveInference, ContinuousVisualizer

agent = ContinuousActiveInference(n_states=2, n_obs=2, n_orders=3, dt=0.05, alpha=0.5, seed=1)
history = {"belief_means": [], "time": [], "actions": [], "free_energy": []}
for _ in range(10):
    action, free_energy = agent.step(np.array([1.0, -1.0]))
    history["belief_means"].append(agent.state.belief_means.copy())
    history["time"].append(agent.state.time)
    history["actions"].append(action)
    history["free_energy"].append(free_energy)

with tempfile.TemporaryDirectory() as tmp:
    viz = ContinuousVisualizer(tmp)
    viz.create_summary_plot(history, Path(tmp) / "summary.png")
    gif = viz.save_animation(history, Path(tmp) / "beliefs.gif", fps=5)
    assert (Path(tmp) / "summary.png").exists() and gif.exists()
```

## Visualisation

`ContinuousVisualizer(output_dir)` provides:

- `plot_belief_evolution(belief_means, belief_precisions, output_path)`
- `plot_generalized_coordinates_relationships(belief_means, time, output_path)`
- `save_animation(history, output_path, fps=30)`, a Pillow GIF with one frame
  per history sample
- `create_summary_plot(history, output_path)`, a 2 by 2 figure of beliefs,
  phase space (first two states), free energy, and actions; the history needs
  `belief_means`, `actions`, and `free_energy`, and optionally `time`

## References

- Friston, K. J. et al. (2017). Active inference, curiosity and insight.
- Buckley, C. L. et al. (2017). The free energy principle for action and
  perception: a mathematical review.
