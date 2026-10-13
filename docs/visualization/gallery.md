---
title: Figure gallery
description: Every figure the trajectory plots and the knowledge-base report draw, what it shows and the command that makes it.
type: reference
status: stable
updated: 2026-10-13
semantic_relations:
  - type: documents
    links:
      - "[[README|Visualization]]"
      - "[[../cli/reference|Command-line reference]]"
      - "[[../config/settings|Settings reference]]"
---

# Figure gallery

Two modules draw figures. `cognitive.visualization.trajectory_plots` draws
simulation results; `scripts/kb_report.py` draws the shape of the knowledge
base. The images below are committed copies in `figures/`; the
commands show how to make fresh ones.

## Simulation figures

Each `plot_*` function takes recorded data and returns a Matplotlib `Figure`.
It writes nothing and leaves the global backend alone, so you decide when to
show, save or close it. `save_figure` writes a figure and closes it.

| Function | Input | Shows |
|---|---|---|
| `plot_trajectory` | a `Trajectory` | Beliefs, choices, free energy and action posterior of one episode. |
| `plot_efe_components` | a `Trajectory` and a `step` | Risk, ambiguity, information gain and the total expected free energy of each action at one step. |
| `plot_sweep` | the summary rows of a sweep | One panel per metric against the swept value, with a band for the spread over seeds. |
| `plot_model` | a `DiscreteGenerativeModel` | The likelihood A, one transition matrix per action, the preferences C and the prior D. |
| `save_figure` | a figure and a path | Writes the figure (`dpi` defaults to 150) and closes it. |

Colour carries the same meaning in every figure: beliefs and probabilities use
the sequential `viridis` map, risk, ambiguity and information gain have fixed
colours, and the agent's own choices are drawn in one orange accent.

### `plot_trajectory`

![Episode of 30 steps on the two-state model](figures/episode.png)

An episode of the two-state model (seed 7, 30 steps). Top left: the agent's
belief over the hidden state at each step, with the true state as a white dot.
Top right: each action chosen and observation received, against the preferred
observation. Bottom left: variational free energy per step, split into
complexity and negative accuracy. Bottom right: the action posterior, with the
chosen action marked. Free energy jumps whenever an observation contradicts the
belief and decays while observations agree with it.

```bash
cognitive simulate --steps 30 --seed 7
```

### `plot_efe_components`

![Expected free energy parts for each action at step 0](figures/efe_components.png)

For each action at one step, four bars: risk (how far the predicted
observations are from the preferred ones), ambiguity (how uncertain the
observations are), the expected free energy G, and information gain (the legend
marks it as included in ambiguity). The action the agent chose is labelled in
the accent colour. At step 0 of the two-state model the two actions have the
same parts, because the prior over states is flat and the actions are
symmetric; the parts separate once beliefs differ.

```bash
cognitive simulate --steps 30 --seed 7   # writes efe_step0.png
```

### `plot_sweep`

![Sweep of temperature, ten episodes per value](figures/temperature_sweep.png)

A sweep of the planning temperature (0.1, 0.3, 1 and 3) with ten episodes per
value. The line is the mean over seeds and the band is the spread. Lower
temperature makes action choice greedier, and here the preference rate falls as
temperature rises while belief accuracy barely moves. Non-numeric values, such
as `policy_aggregation`, are drawn on a categorical axis.

```bash
cognitive sweep --parameter temperature --values 0.1,0.3,1,3 --seeds 10 --steps 30
```

### `plot_model`

![The generative model matrices](figures/model.png)

The model's matrices as heat maps: A maps states to observations, each B maps
the previous state to the next under one action, C holds the log preferences
over observations and D the prior over states.

```bash
cognitive simulate --steps 30 --seed 7   # writes model.png
```

## Knowledge-base figures

`cognitive-kb-report` reads the RDF export back into a directed graph and writes
`report.json`, `report.md` and four figures. The report is derived data and is
not committed; the three figures below are committed so that this page renders.

```bash
cognitive-kb-report                      # figures, JSON and Markdown in output/kb_report
cognitive-kb-report --no-figures         # numbers only
cognitive-kb-report --format svg --dpi 200
```

| Figure file | Shows |
|---|---|
| `in_degree_distribution` | Share of pages with at least k incoming links, on log axes. A long tail means a few hub pages and many pages with few links. |
| `domain_sizes` | Pages per domain, with the share of each domain's links that stay inside it. |
| `domain_links` | Links between domains, from row to column, on a log colour scale. |
| `relations` | Edges per relation name, on a log scale (the 20 most common). |

### `domain_sizes`

![Pages per domain](figures/kb_domain_sizes.png)

Pages per domain. The `cognitive` and `mathematics` domains hold most pages,
and 84% of their links stay inside. The small `ontology`, `BioFirm` and
top-level groups link mostly outward.

### `in_degree_distribution`

![Incoming links per page](figures/kb_in_degree.png)

Each dot is the share of pages with at least k incoming links. The share falls
steadily and then drops off sharply: about one page in ten has more than 20
incoming links, about one in a hundred has more than 80, and the most linked
page has over 300. The graph has one connected component and no isolated page.

### `domain_links`

![Links between domains](figures/kb_domain_links.png)

Each cell is the number of links from the row domain to the column domain. The
bright diagonal is links inside a domain; off-diagonal cells show which domains
refer to each other.

The `relations` figure is written by the command but not committed here.

## Adding a figure

1. Write a `plot_*` function in `trajectory_plots.py` that takes data and
   returns a `Figure`.
2. Add a test in `code/tests/test_trajectory_plots.py`.
3. Describe the function on this page. `code/tests/test_docs_sync.py` fails when
   a public `plot_*` or `save_*` function is missing here, or when an image link
   points at a file that does not exist.
