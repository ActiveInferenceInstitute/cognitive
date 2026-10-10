---
title: API integration examples
type: guide
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[docs/api/api_reference]]"
      - "[[docs/examples/basic_api_usage]]"
---

# API integration examples

The package integrates with three external pieces: NumPy arrays for all
model data, YAML files for strict configuration, and matplotlib/seaborn for
figures. PyTorch and other machine-learning frameworks are not dependencies and
no integration for them exists.

## YAML configuration

`ActiveInferenceFactory.create_from_yaml` reads an inference section and a
`generative_model` section from one file. Unknown keys raise `ValueError`. The
shipped file is
`code/tools/src/models/active_inference/config_examples/discrete_variational.yaml`.

```python
from pathlib import Path

from cognitive import ActiveInferenceFactory

path = Path("code/tools/src/models/active_inference/config_examples/discrete_variational.yaml")
dispatcher = ActiveInferenceFactory.create_from_yaml(path)
print(dispatcher.model.num_states, dispatcher.config.method.value)
```

Run it from the repository root.

## Plotting a model matrix

`MatrixPlotter(output_dir, style_config)` wraps seaborn heatmaps and writes the
figure when `save_name` is given.

```python
import tempfile

import matplotlib

matplotlib.use("Agg")

import numpy as np

from cognitive import MatrixPlotter

style = {"theme": "default", "figure_size": (4, 3), "dpi": 100, "colormap": "viridis"}
with tempfile.TemporaryDirectory() as tmp:
    plotter = MatrixPlotter(tmp, style)
    figure = plotter.plot_heatmap(
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        title="A matrix",
        xlabel="state",
        ylabel="observation",
        save_name="likelihood",
    )
```

Output paths are always caller-selected; the package does not assume a
repository output directory. Other helpers are listed in
[[docs/api/api_reference|API reference]].
