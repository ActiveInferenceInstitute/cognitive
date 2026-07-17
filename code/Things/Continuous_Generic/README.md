# Continuous generalized-coordinate agent

`ContinuousActiveInference` performs precision-weighted updates over a state
matrix shaped `(n_states, n_orders)`. The zeroth order is the latent state;
higher orders are generalized derivatives. Configuration can provide an
observation matrix, dynamics matrix, observation precision, state precision,
precision learning rate, and seed.

`ContinuousVisualizer.save_animation` uses Matplotlib `FuncAnimation` and
`PillowWriter`. It validates the history and writes one rendered frame per
sample rather than a fixed image header.

```python
import numpy as np

from Things.Continuous_Generic import ContinuousActiveInference, ContinuousVisualizer

agent = ContinuousActiveInference(n_states=2, n_obs=2, n_orders=3, seed=11)
history = {"belief_means": [], "time": []}
for _ in range(4):
    history["belief_means"].append(agent.state.belief_means.copy())
    history["time"].append(agent.state.time)
    agent.step(np.zeros(2))
ContinuousVisualizer("/tmp/cognitive-visuals").save_animation(
    history, "/tmp/cognitive-visuals/beliefs.gif", fps=10
)
```
