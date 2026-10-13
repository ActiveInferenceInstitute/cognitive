---
title: Visualization documentation instructions
description: Rules for editing the visualization documentation and its committed figures.
type: agents
status: stable
updated: 2026-10-13
semantic_relations:
  - type: documents
    links:
      - "[[README|Visualization]]"
      - "[[gallery|Figure gallery]]"
---

# Visualization documentation instructions

## Ground truth

- Simulation figures: `code/tools/src/visualization/trajectory_plots.py`. Every
  public `plot_*` and `save_*` function must be named in
  [`gallery.md`](gallery.md).
- Knowledge-base figures: `draw_figures` in `code/scripts/kb_report.py`. The
  gallery names its four figure files.
- `figures/` holds committed copies for the pages. They are small, regenerable
  and never the source of a number: describe a figure from the data behind it
  (`report.json`, `trajectory.json`), not only from the picture.

## Rules

1. Look at a figure before describing it. Describe what is drawn, not what it
   was meant to show.
2. Give the command that regenerates each figure, and keep the arguments in the
   caption equal to the ones used.
3. Do not commit generated output other than the images the gallery links.
4. Reference images with a relative `figures/<name>.png` link; the docs-sync test
   checks that each linked file exists.

## Verification

```bash
python -m pytest code/tests/test_docs_sync.py code/tests/test_trajectory_plots.py -q
```
