---
title: Learning Visualization
type: guide
status: stable
created: 2025-03-03
updated: 2026-10-09
tags:
- visualization
- learning
semantic_relations:
  - type: relates_to
    links:
      - "[[worked_example_vfe]]"
      - "[[exploration_visualization]]"
      - "[[convergence_control]]"
---


# Learning Visualization

Displays for VFE/EFE histories, component balances, and convergence trends. The goal is to see, from a run, whether belief updating is working: whether free energy falls when evidence agrees with the beliefs, spikes when it does not, and whether the optimiser has converged.

## What a free-energy history means

For exact categorical inference with a belief that carries over between steps, the variational free energy at each step equals the surprise of that observation given the past, $F_t = -\ln p(o_t \mid o_{<t})$, because the KL term to the exact posterior is zero ([[variational_free_energy]]). So a $F_t$ trace is a surprise trace: low when the data fit the current belief, high when they contradict it, and not expected to decrease monotonically.

Take the two-state model of [[worked_example_vfe]] with an identity transition and the observation sequence $0, 0, 1, 0, 0, 1, 1, 1$. The trace of $F_t$ is

$$0.478,\ 0.211,\ 2.101,\ 0.282,\ 0.149,\ 2.217,\ 1.796,\ 0.872 \text{ nats}$$

and the belief in state 0 moves $0.6 \to 0.871 \to 0.968 \to 0.792 \to 0.945 \to 0.987 \to 0.906 \to 0.546 \to 0.131$. The two spikes at steps 3 and 6 are the observations of outcome 1 while the agent was confident in state 0. The long run of outcome 1 at the end drives the belief across 0.5, and $F_t$ decays as the new belief becomes right.

## Component balance

Splitting $F_t$ into $-\text{accuracy}$ and complexity shows what the update paid for. At step 3 the accuracy term is $-1.869$ and the complexity $0.232$: almost all of the free energy is poor fit, and the belief revision is cheap. At step 7 complexity is $0.438$, larger than at any other step, because the belief moves furthest from its prior there. A bar stack of the two terms with $F_t$ overlaid makes both visible; the identity $F = \text{complexity} - \text{accuracy}$ gives a built-in check.

## Convergence of the optimiser

Inside one step, plot $F$ against iteration number of whatever minimiser is used. For gradient or mirror-descent updates ([[optimization_strategies]]) the curve should be non-increasing for a small enough step; a rise signals a step size that is too large. A flat curve at a value above $-\ln p(o)$ in a model where that is computable signals that the variational family cannot match the posterior ([[variational_bounds]]). Stopping criteria are discussed in [[convergence_control]].

## Expected free energy histories

Plot the per-action values from `ActiveInferenceDispatcher.calculate_expected_free_energy(state)` over time, or the chosen policy's $G$ and its risk and ambiguity components from `DiscreteGenerativeModel.expected_free_energy`. A shrinking ambiguity component with stable risk is the signature of an agent that has resolved its uncertainty and is now exploiting ([[exploration_visualization]]). See [[expected_free_energy]] for the definitions.

## Tools in the repository

- `MatrixPlotter.plot_belief_evolution(beliefs, state_labels=...)` draws one line per state from an array shaped (time, states) and returns the axes.
- `StateSpacePlotter.plot_belief_evolution(beliefs, title, state_labels, save_name)` is the figure-returning variant with markers.
- `MatrixPlotter.save_figure(fig, name)` writes into the plotter's `output_dir`.
- Free-energy traces, component stacks and iteration curves are plain matplotlib on arrays you collect in the loop; no dedicated plotting function exists for them in `code/tools/src/visualization/matrix_plots.py`.
- `ModelState.free_energy`, updated by `ActiveInferenceModel.calculate_free_energy`, is a simpler proxy (half the mean squared prediction error plus a KL term), so recompute the true $F$ as below when you need the numbers on this page.

## Reproducing the numbers

```python
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cognitive.models.active_inference import DiscreteGenerativeModel
from cognitive.visualization.matrix_plots import MatrixPlotter

A = np.array([[0.9, 0.2], [0.1, 0.8]])
model = DiscreteGenerativeModel(A, np.eye(2)[:, :, None], np.zeros(2),
                                np.array([0.6, 0.4]), np.ones(1))

observations = [0, 0, 1, 0, 0, 1, 1, 1]
beliefs = model.D.copy()
history, free_energy, accuracy, complexity = [beliefs], [], [], []
for o in observations:
    prior = model.predict_states(beliefs, 0)
    posterior = model.posterior(o, prior)
    acc = float(np.sum(posterior * np.log(model.A[o])))
    cx = float(np.sum(posterior * (np.log(posterior) - np.log(prior))))
    accuracy.append(acc)
    complexity.append(cx)
    free_energy.append(cx - acc)
    beliefs = posterior
    history.append(beliefs)

print("F       ", np.round(free_energy, 4))
print("accuracy", np.round(accuracy, 4))
print("complexity", np.round(complexity, 4))
print("belief in s0", np.round(np.array(history)[:, 0], 3))

out = Path(tempfile.mkdtemp())
plotter = MatrixPlotter(str(out), {"figure_size": (6, 4), "dpi": 100})

ax = plotter.plot_belief_evolution(np.array(history), state_labels=["s0", "s1"])
plotter.save_figure(ax.figure, "belief_evolution")

fig, ax = plt.subplots(figsize=(6, 4))
steps = np.arange(len(observations))
ax.bar(steps, -np.array(accuracy), label="-accuracy")
ax.bar(steps, complexity, bottom=-np.array(accuracy), label="complexity")
ax.plot(steps, free_energy, "k.", label="F")
ax.set_xlabel("step")
ax.set_ylabel("nats")
ax.legend()
plotter.save_figure(fig, "vfe_components")
print(sorted(p.name for p in out.iterdir()))
```

## Links

- [[vfe_components]]
- [[expected_free_energy]]
- [[worked_example_vfe]] · [[exploration_visualization]]
