---
title: Continuous generalized-coordinate agent
type: reference
status: stable
updated: 2026-10-10
semantic_relations:
  - type: documented_by
    links:
      - "[[code/Things/Continuous_Generic/Continuous_Generic_README]]"
      - "[[code/Things/Continuous_Generic/AGENTS]]"
      - "[[docs/api/api_reference|API reference]]"
---

# Continuous generalized-coordinate agent

`ContinuousActiveInference` (`code/Things/Continuous_Generic/continuous_generic.py`)
performs precision-weighted updates over a state matrix shaped
`(n_states, n_orders)`. The zeroth order is the latent state; higher orders are
generalized derivatives. Constructor arguments: `n_states`, `n_obs`, `n_orders`,
`dt`, `alpha`, `observation_matrix`, `dynamics_matrix`, `observation_precision`,
`state_precision`, `precision_learning_rate`, `dynamics_weight`, `seed`, and the
bounds `precision_min` and `precision_max`.

`step(observation)` returns the action and the free energy; `reset` restores the
beliefs. `ContinuousVisualizer.save_animation` uses Matplotlib `FuncAnimation`
and `PillowWriter` and writes one rendered frame per history sample.

```python
import tempfile
from pathlib import Path

import numpy as np

from Things.Continuous_Generic import ContinuousActiveInference, ContinuousVisualizer

agent = ContinuousActiveInference(n_states=2, n_obs=2, n_orders=3, seed=11)
history = {"belief_means": [], "time": []}
for _ in range(4):
    history["belief_means"].append(agent.state.belief_means.copy())
    history["time"].append(agent.state.time)
    agent.step(np.zeros(2))

out = Path(tempfile.mkdtemp())
ContinuousVisualizer(out).save_animation(history, out / "beliefs.gif", fps=10)
```
