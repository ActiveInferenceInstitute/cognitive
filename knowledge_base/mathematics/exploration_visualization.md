---
title: Exploration Visualization
type: guide
status: stable
created: 2025-03-03
updated: 2026-10-09
tags:
- visualization
- exploration
semantic_relations:
  - type: relates_to
    links:
      - "[[exploration_strategies]]"
      - "[[mutual_information]]"
      - "[[learning_visualization]]"
---


# Exploration Visualization

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

Visual tools for epistemic value, uncertainty maps, and exploration paths. Exploration in Active Inference is driven by the expected information gain $I(s;o)$ inside expected free energy ([[exploration_strategies]]), so the useful plots are those that show where that gain is large, how it changes with belief, and how it compares with the pragmatic term.

## What to plot

**Information gain per action.** For the current belief, one bar per action of $I(s;o)$. This shows which action the epistemic drive favours, independently of preferences. `DiscreteGenerativeModel.expected_free_energy(beliefs, action)` returns it as the fourth element of its tuple. For the toy model of [[worked_example_efe]] at beliefs $(0.3, 0.7)$ the bars are $0.0813$ for "stay" and $0.0431$ for "move" nats.

**Information gain against belief.** Sweep $q(s_0)$ from 0 to 1 and plot the gain for one action. It is zero at deterministic beliefs and peaks at an intermediate belief, which shows why exploration switches itself off. For "stay" in the toy model the curve peaks at about $0.102$ nats near $q(s_0) = 0.53$ (50-point grid). A shifted peak, rather than one at 0.5, reflects that state 1 is ambiguous and state 0 is not.

**Likelihood heatmap.** A heatmap of $A[o, s]$ shows which states are informative (columns far from uniform) and which are ambiguous (columns near uniform). The ambiguity term is the expected entropy of these columns.

**Policy values.** Bars of $G(\pi)$ for each candidate policy, so exploratory and exploitative choices can be compared with the pragmatic part visible.

## Tools that exist in the repository

| need | code | notes |
|------|------|-------|
| bar chart of a vector | `MatrixPlotter.plot_bar(values, title=, xlabel=, ylabel=, save_name=)` | in `code/tools/src/visualization/matrix_plots.py`; returns the figure, and `save_name` writes into `output_dir` and closes it |
| matrix heatmap | `MatrixPlotter.plot_heatmap(matrix, ..., annot=True)` | extra keywords go to `seaborn.heatmap` |
| 3-state free-energy surface | `MatrixPlotter.plot_free_energy_landscape(func, ...)` | `func` receives a 3-vector of beliefs; the z-axis is labelled expected free energy |
| belief trajectories | `MatrixPlotter.plot_belief_evolution`, `StateSpacePlotter.plot_belief_evolution` | beliefs shaped (time, states) |
| policy values | `StateSpacePlotter.plot_policy_evaluation(values, labels, title, save_name)` | needs a `save_dir` to write |
| any other figure | build with matplotlib, then `MatrixPlotter.save_figure(fig, name)` | uses `style_config["file_format"]` and dpi |

The knowledge-graph renderer `NetworkVisualizer` in `code/tools/src/utils/visualization/network_viz.py` draws the link structure of this vault, not agent behaviour, and is unrelated to exploration plots. There is no dedicated exploration-plotting module, so the information-gain curve is plain matplotlib.

## Example

```python
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from cognitive.models.active_inference import DiscreteGenerativeModel
from cognitive.visualization.matrix_plots import MatrixPlotter, StateSpacePlotter

A = np.array([[0.9, 0.5], [0.1, 0.5]])
B = np.zeros((2, 2, 2))
B[:, :, 0] = np.eye(2)
B[:, :, 1] = [[0.9, 0.9], [0.1, 0.1]]
model = DiscreteGenerativeModel(A, B, np.array([2.0, 0.0]), np.full(2, 0.5), np.full(2, 0.5))
beliefs = np.array([0.3, 0.7])

out = Path(tempfile.mkdtemp())
plotter = MatrixPlotter(str(out), {"figure_size": (6, 4), "dpi": 100})
epistemic = [model.expected_free_energy(beliefs, a)[3] for a in range(model.num_actions)]
print(epistemic)
plotter.plot_bar(epistemic, title="Epistemic value I(s; o) per action",
                 xlabel="action", ylabel="nats", save_name="epistemic_by_action")
plotter.plot_heatmap(A, title="Likelihood A[o, s]", xlabel="state", ylabel="observation",
                     annot=True, save_name="likelihood")

# belief-entropy surface: epistemic value as a function of the belief q(s_0)
grid = np.linspace(0.01, 0.99, 50)
gain = [model.expected_free_energy(np.array([b, 1 - b]), 0)[3] for b in grid]
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(grid, gain)
ax.set_xlabel("q(s=0)"); ax.set_ylabel("expected information gain (nats)")
plotter.save_figure(fig, "gain_vs_belief")

state_plotter = StateSpacePlotter(out)
policies = model.enumerate_policies(2)
values = [model.evaluate_policy(beliefs, p, 1.0) for p in policies]
state_plotter.plot_policy_evaluation(np.array(values), [str(p) for p in policies],
                                     "Policy evaluation", save_name="policies")
print(sorted(p.name for p in out.iterdir()))
print(max(gain), grid[int(np.argmax(gain))])
```

The first print gives the two bars. The second gives the file names written, and the last the peak of the sweep. Check any plot against a number you can compute by hand before interpreting it.

## Reading the plots

- Flat information gain across actions means preferences, not curiosity, decide the choice.
- Information gain near zero everywhere with a still-uncertain belief means the observations are uninformative about the states: look at the ambiguity of $A$ rather than the policy.
- If the peak of the gain curve sits where the agent never goes, check the preference vector $C$ and the horizon; pragmatic value may dominate.

## Links

- [[epistemic_value]]
- [[exploration_exploitation]]
- [[exploration_strategies]] · [[learning_visualization]]
