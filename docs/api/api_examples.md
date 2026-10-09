---
title: API examples
type: reference
status: stable
updated: 2026-10-09
---

# API examples

Each snippet was run against the current tree with the editable install. The
dispatcher example is in [`api_reference.md`](api_reference.md); more executable
examples live in [`../examples/README.md`](../examples/README.md).

## Simple POMDP

```python
import tempfile
from pathlib import Path

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
    results = agent.run(5)  # list of (observation, free_energy)
    assert len(results) == 5 and agent.state.time_step == 5

    saved = agent.save_state(Path(tmp) / "state.yaml")
    restored = SimplePOMDP(config)
    restored.load_state(saved)
    assert restored.state.time_step == 5
```

## Continuous agent and animation

```python
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from Things.Continuous_Generic import ContinuousActiveInference, ContinuousVisualizer

agent = ContinuousActiveInference(n_states=2, n_obs=2, n_orders=3, dt=0.05, alpha=0.5, seed=1)
history = {"belief_means": [], "time": []}
for _ in range(5):
    action, free_energy = agent.step(np.array([1.0, -1.0]))
    history["belief_means"].append(agent.state.belief_means.copy())
    history["time"].append(agent.state.time)

with tempfile.TemporaryDirectory() as tmp:
    gif = ContinuousVisualizer(tmp).save_animation(history, Path(tmp) / "beliefs.gif", fps=5)
    assert gif.exists()
```

## Homeostatic agent

```python
import matplotlib

matplotlib.use("Agg")

from cognitive import HomeostaticFactory

identity = [[1.0, 0.0], [0.0, 1.0]]
config = {
    "state_spaces": {
        "environment": {
            "dimensions": [2],
            "labels": {"states": ["LOW", "HIGH"]},
            "mappings": {"identity": identity},
        },
        "observation": {
            "dimensions": [2],
            "labels": {"observations": ["low", "high"]},
            "mappings": {"identity": identity},
        },
        "action": {
            "dimensions": [2],
            "labels": {"actions": ["stay", "switch"]},
            "mappings": {"identity": identity},
        },
    },
    "observation_model": {"likelihood_matrix": [[0.9, 0.1], [0.1, 0.9]]},
    "transition_model": {
        "transition_matrices": {"stay": identity, "switch": [[0.1, 0.9], [0.9, 0.1]]},
        "temporal_horizon": 1,
    },
    "inference": {
        "method": "variational",
        "policy_type": "discrete",
        "temporal_horizon": 1,
        "learning_rate": 0.2,
        "precision_init": 1.0,
    },
    "target_state": [0.8, 0.2],
    "initial_beliefs": [0.5, 0.5],
}
agent = HomeostaticFactory.create_basic(config)
observation, free_energy = agent.step()
assert abs(agent.state.beliefs.sum() - 1.0) < 1e-9
```

## Probability helpers

```python
import numpy as np

from cognitive import compute_entropy, kl_divergence, softmax
from cognitive.models.matrices.matrix_ops import MatrixInitializer, MatrixOps

p = softmax(np.array([0.0, 1.0, 2.0]))
assert np.isclose(p.sum(), 1.0)
divergence = kl_divergence(p, np.full(3, 1 / 3))
entropy = compute_entropy(p)

columns = MatrixOps.normalize_columns(np.array([[1.0, 2.0], [3.0, 2.0]]))
assert np.allclose(columns.sum(axis=0), 1.0)
uniform = MatrixInitializer.uniform((2, 2))
```
