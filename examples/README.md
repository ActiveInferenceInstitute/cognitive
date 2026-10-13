---
title: Examples
description: Runnable discrete, continuous and hierarchical episodes, sweeps and knowledge-base reports.
type: documentation
status: stable
updated: 2026-10-13
semantic_relations:
  - type: documents
    links:
      - "[[AGENTS|Examples instructions]]"
      - "[[../docs/cli/reference|Command-line reference]]"
      - "[[../docs/config/settings|Settings reference]]"
      - "[[../docs/visualization/gallery|Figure gallery]]"
      - "[[../docs/examples/README|Example pages]]"
---

# Examples

Eight short scripts, each a complete run. Every script is a thin layer over the
package: it loads settings, calls the library and writes its results to
`output/examples/<name>/` (change that with `--output`). Run them from the
repository root after `pip install -e .`:

```bash
python examples/01_run_episode.py
```

| Script | What it shows | Writes |
|---|---|---|
| [`01_run_episode.py`](01_run_episode.py) | One episode of the two-state model with the settings and model in `configs/`. | `trajectory.json`, `episode.png`, `efe_step0.png` |
| [`02_compare_aggregation.py`](02_compare_aggregation.py) | How policies that share a first action are combined (`marginal` or `min`), on a corridor with a goal. `--seeds` sets the episodes per value (default 20). | `summary.csv`, `aggregation.png` |
| [`03_sensor_noise_sweep.py`](03_sensor_noise_sweep.py) | How a noisier sensor changes what the agent believes and how often it gets what it prefers. `--seeds` default 10. | `summary.csv`, `sensor_noise.png` |
| [`04_model_mismatch.py`](04_model_mismatch.py) | An agent whose model is wrong: believed dynamics against real dynamics. | `correct_model.png`, `wrong_model.png` |
| [`05_knowledge_base_report.py`](05_knowledge_base_report.py) | How the knowledge base is connected: domain sizes, links, hubs. | `report.json`, `report.md` and four figures |
| [`06_export_rdf.py`](06_export_rdf.py) | The knowledge base as RDF (N-Quads and Turtle) for the Open Knowledge Network. | `cognitive-kb.nq`, `cognitive-kb.ttl` |
| [`07_continuous_episode.py`](07_continuous_episode.py) | A finite generalized-coordinate agent controlling an Euler world with a noisy sensor. | `trajectory.json`, `trajectory.csv`, `episode.png` |
| [`08_hierarchical_episode.py`](08_hierarchical_episode.py) | An upper goal selector and a lower corridor controller operating at different timescales. | `hierarchy.json`, `trajectory.json`, `episode.png` |

Scripts 05 and 06 take `--root` (default: this repository) and run on the real
knowledge base.

## Configuration files

| File | Used by | Contents |
|---|---|---|
| [`configs/settings.yaml`](configs/settings.yaml) | 01 | A longer episode (40 steps, seed 7), a shorter planning horizon and a lower figure resolution. Sections not listed keep their defaults. |
| [`configs/two_state.yaml`](configs/two_state.yaml) | 01 | A light that is dark or light, a sensor that is right 85% of the time, an agent that prefers to see "bright". |

Scripts 02 to 04 use the built-in defaults with `environ={}`, so that
`COGNITIVE_*` variables on your machine cannot change their results.

## Adapt one

- Change the episode: edit `configs/settings.yaml`, or copy it and pass the copy
  to `cognitive simulate --config`.
- Use your own model: copy `configs/two_state.yaml`, change the matrices and run
  `cognitive simulate --model your_model.yaml`.
- Sweep another parameter: `cognitive sweep --parameter temperature
  --values 0.3,1,3`. On the built-in two-state model the planning horizon
  (`temporal_horizon`) gives identical results, because its two actions are
  symmetric; use `cognitive sweep --builtin corridor --parameter temporal_horizon
  --values 1,2,3` to study planning.

The scripts are tested: `code/tests/test_examples.py` runs every one and checks
the files it writes. For the command-line equivalents see
[`docs/cli/reference.md`](../docs/cli/reference.md); for what each figure shows
see [`docs/visualization/gallery.md`](../docs/visualization/gallery.md).
