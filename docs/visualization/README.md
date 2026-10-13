---
title: Visualization
description: Figures for simulation episodes, expected free energy, parameter sweeps, generative models and the connectivity of the knowledge base.
type: documentation
status: stable
updated: 2026-10-13
semantic_relations:
  - type: organizes
    links:
      - "[[gallery|Figure gallery]]"
      - "[[AGENTS|Visualization documentation instructions]]"
      - "[[../cli/reference|Command-line reference]]"
      - "[[../README|Documentation hub]]"
---

# Visualization

Two sources of figures, one catalogue:

- `cognitive.visualization.trajectory_plots` draws what happens in a
  simulation: one episode, the expected free energy of each action, a sweep of
  a parameter, the matrices of a model, and continuous and hierarchical episodes.
- `scripts/kb_report.py` (`cognitive-kb-report`) draws how the knowledge base is
  connected: domain sizes, links between domains, incoming links per page and
  edge counts by relation.

[`gallery.md`](gallery.md) shows every figure, says how to read it and gives the
command that makes it. The committed images live in `figures/`.

## Use from Python

```python
import matplotlib

matplotlib.use("Agg")
from cognitive.settings import load_settings
from cognitive.simulation import example_model, run_episode
from cognitive.visualization.trajectory_plots import plot_trajectory, save_figure

settings = load_settings(environ={})
trajectory = run_episode(example_model(), settings.inference_config(), steps=30, seed=7)
save_figure(plot_trajectory(trajectory), "episode.png", dpi=settings.output.dpi)
```

The `plot_*` functions return a `Figure` and do not choose a backend, so the
same call works in a notebook and in a script.

## Use from the shell

```bash
cognitive simulate --steps 30 --seed 7     # episode, efe_step0 and model figures
cognitive sweep --parameter temperature --values 0.1,0.3,1,3
cognitive-kb-report                         # knowledge-base figures
uv run python -m scripts.regenerate_gallery --publish  # all figures and captions
```

`output.figure_format` (`png`, `svg` or `pdf`) and `output.dpi` choose the file
type and resolution; see the [settings reference](../config/settings.md).

## See also

- [`../cli/reference.md`](../cli/reference.md): files each command writes.
- [`../../examples/README.md`](../../examples/README.md): scripts that draw these figures.
