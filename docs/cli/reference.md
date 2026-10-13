---
title: Command-line reference
description: Every command, option, output file and exit status of the cognitive command, plus the report and export commands.
type: reference
status: stable
updated: 2026-10-13
semantic_relations:
  - type: documents
    links:
      - "[[README|Command line]]"
      - "[[../config/settings|Settings reference]]"
      - "[[../visualization/gallery|Figure gallery]]"
---

# Command-line reference

`pip install -e .` installs the `cognitive` command. `python -m cognitive` runs
the same program, which is useful in a checkout that is not installed.

```text
cognitive {settings,simulate,sweep,benchmark} ...
```

Every option that sets a value is resolved through the layers described in the
[settings reference](../config/settings.md): built-in defaults, then the YAML
file, then `COGNITIVE_*` environment variables, then the options below.

## Options shared by `settings`, `simulate` and `sweep`

| Option | Meaning |
|---|---|
| `--config CONFIG` | YAML settings file. Default: `./config.yaml` if it exists. A path that is given must exist. |
| `--set SECTION.KEY=VALUE` | Override one setting, for example `simulation.steps=50`. Repeatable. |
| `--json` | Print JSON instead of text. |

`simulate` and `sweep` also take:

| Option | Meaning |
|---|---|
| `--model MODEL` | YAML generative model. Default: the built-in two-state model. |
| `--output OUTPUT` | Directory for the result files. Default: `output.directory` (`output`). |
| `--no-figures` | Write tables and JSON only. |

## `cognitive settings`

Prints the settings after every layer is applied. Nothing is written.

```bash
cognitive settings
cognitive settings --config examples/configs/settings.yaml --set simulation.steps=50 --json
```

## `cognitive simulate`

Runs one episode: a hidden state evolves under the agent's actions, the agent
updates its beliefs and picks each action by minimizing expected free energy.

```text
cognitive simulate [--config CONFIG] [--set SECTION.KEY=VALUE] [--model MODEL]
                   [--output OUTPUT] [--no-figures] [--json]
                   [--steps STEPS] [--seed SEED]
```

| Option | Meaning |
|---|---|
| `--steps STEPS` | Shorthand for `--set simulation.steps=STEPS`. Wins over `--set`. |
| `--seed SEED` | Shorthand for `--set simulation.seed=SEED`. Wins over `--set`. |

The same seed, model and settings give the same episode.

```bash
cognitive simulate --steps 30 --seed 7 --output output/episode
cognitive simulate --model examples/configs/two_state.yaml --no-figures --json
```

Files written to the output directory:

| File | Contents |
|---|---|
| `trajectory.json` | The whole episode: configuration, labels, states, observations, actions, beliefs, free energy and its parts, policy posteriors, expected free energy. |
| `trajectory.csv` | One row per step with the scalars only: `step`, `state`, `action`, `observation`, `free_energy`, `complexity`, `accuracy`. Beliefs and posteriors are in the JSON. |
| `episode.<format>` | Four panels: beliefs, choices and observations, free energy, action posterior ([`plot_trajectory`](../visualization/gallery.md)). |
| `efe_step0.<format>` | Expected free energy of each action at the first step ([`plot_efe_components`](../visualization/gallery.md)). |
| `model.<format>` | The model's matrices ([`plot_model`](../visualization/gallery.md)). |

`<format>` is `output.figure_format`. The three figures are skipped with
`--no-figures`.

Printed summary (also under `summary` in the JSON): `steps`, `seed`,
`mean_free_energy`, `mean_complexity`, `mean_accuracy`, `belief_accuracy` (mean
probability given to the true state after each update), `preference_rate`
(fraction of observations equal to the preferred one), `mean_policy_entropy`
and `action_counts`.

## `cognitive sweep`

Varies one inference parameter over several values and runs several seeds at
each.

```text
cognitive sweep --parameter PARAMETER --values VALUES [--seeds SEEDS]
                [--steps STEPS] [--config CONFIG] [--set SECTION.KEY=VALUE]
                [--model MODEL] [--output OUTPUT] [--no-figures] [--json]
```

| Option | Meaning |
|---|---|
| `--parameter PARAMETER` | Required. One of `discount_factor`, `exploration_weight`, `learning_rate`, `method`, `policy_aggregation`, `precision_init`, `temperature`, `temporal_horizon`. |
| `--values VALUES` | Required. Comma-separated values to try. Each is converted to the parameter's type, so `--values 1,2,3` is valid for `temporal_horizon` and `--values marginal,min` for `policy_aggregation`. |
| `--seeds SEEDS` | Episodes per value (default 5). Seeds run from `simulation.seed` upward. |
| `--steps STEPS` | Shorthand for `--set simulation.steps=STEPS`. |

```bash
cognitive sweep --parameter temperature --values 0.1,0.3,1,3 --seeds 10 --steps 30
cognitive sweep --parameter policy_aggregation --values marginal,min --no-figures
```

Files written:

| File | Contents |
|---|---|
| `sweep_episodes.csv` | One row per episode: `parameter`, `value`, `seed`, `steps` and the scalar episode statistics. |
| `sweep_summary.csv` | One row per value, with the number of `episodes` and, for each metric, its mean over the seeds and its standard deviation (columns ending `_std`). |
| `sweep.<format>` | One panel per metric against the swept value, with a band for the spread across seeds ([`plot_sweep`](../visualization/gallery.md)). |

Each text line shows `preference_rate`, `belief_accuracy` and
`mean_free_energy` for one value.

The built-in model has two symmetric actions, so on it `temporal_horizon` gives
identical results at every value; sweep `temperature` or another parameter, or
use a model with asymmetric actions (see
[`examples/02_compare_aggregation.py`](../../examples/README.md)).

## `cognitive benchmark`

Times the runtime components. Everything after `benchmark` goes to the
benchmark program, so its options are its own:

```text
cognitive benchmark [-h] [--repetitions REPETITIONS] [--output OUTPUT]
```

`--output` writes the JSON result to a file as well as printing it. The
`cognitive-benchmark` command is the same program.

## Exit status

| Status | Meaning |
|---|---|
| `0` | Success. |
| `2` | A setting, file or argument was rejected. The message is printed to stderr as `error: ...`: an unknown settings key, a value of the wrong type or out of range, a missing `--config` or `--model` file, an empty `--values`, `--seeds` below 1. |

Mistakes in the command line itself (an unknown option or a missing required
option) are reported by `argparse` and also exit with status 2.

## Report and export commands

These are separate commands installed with the package. They work on the
knowledge base rather than on a simulation.

| Command | Purpose |
|---|---|
| `cognitive-kb-report [ROOT] [--output DIR] [--format png\|svg\|pdf] [--dpi N] [--no-figures]` | Measures how the knowledge base is connected and writes `report.json`, `report.md` and four figures to `output/kb_report`. It reads the graph back from the RDF export, so it reports what the published graph contains. The report is derived data and is not committed. |
| `cognitive-export-rdf` | Exports the knowledge base as RDF (N-Quads and Turtle) for the Open Knowledge Network. |
| `cognitive-kb-index`, `cognitive-verify-links`, `cognitive-validate-docs`, `cognitive-normalize-frontmatter`, `cognitive-create-node` | Knowledge-base maintenance; see [`../tools/README.md`](../tools/README.md). |

The figures `cognitive-kb-report` writes are described in the
[figure gallery](../visualization/gallery.md). The scripts in
[`examples/`](../../examples/README.md) call both commands.
