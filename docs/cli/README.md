---
title: Command line
description: Run a simulation, a parameter sweep or the benchmarks from the shell with the cognitive command.
type: documentation
status: stable
updated: 2026-10-13
semantic_relations:
  - type: organizes
    links:
      - "[[reference|Command-line reference]]"
      - "[[AGENTS|Command-line documentation instructions]]"
      - "[[../config/settings|Settings reference]]"
      - "[[../visualization/README|Visualization]]"
      - "[[../README|Documentation hub]]"
---

# Command line

The `cognitive` command runs the package's simulation without writing any
Python. It provides these subcommands:

| Command | What it does |
|---|---|
| `cognitive settings` | Print the settings a run would use. |
| `cognitive simulate` | Run one episode and write its trajectory and figures. |
| `cognitive sweep` | Vary one inference parameter over several values and seeds. |
| `cognitive benchmark` | Time the runtime components. |
| `cognitive plot` | Redraw a saved discrete trajectory. |
| `cognitive continuous` | Run a continuous agent in a controlled Euler world. |
| `cognitive hierarchical` | Run upper goal selection and lower corridor control. |

## A first run

From the repository root, after `pip install -e .`:

```bash
cognitive simulate --steps 30 --seed 7 --output output/episode
```

This prints the episode summary and writes `trajectory.json`, `trajectory.csv`
and three figures to `output/episode/`. Then compare how the agent behaves at
different planning temperatures:

```bash
cognitive sweep --parameter temperature --values 0.1,0.3,1,3 --seeds 10 --steps 30
```

Every number the command uses can be changed in `config.yaml`, in an
environment variable or on the command line; `cognitive settings` shows the
result. The [settings reference](../config/settings.md) lists every key.

## Where to go next

- [`reference.md`](reference.md): every option, output file and exit status.
- [`../visualization/gallery.md`](../visualization/gallery.md): what each figure shows.
- [`../../examples/README.md`](../../examples/README.md): scripts that do the same from Python.
- [`../architecture/README.md`](../architecture/README.md): how the command is built.
