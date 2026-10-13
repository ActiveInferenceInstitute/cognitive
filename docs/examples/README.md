---
title: Executable examples
type: documentation
status: stable
updated: 2026-10-13
semantic_relations:
  - type: relates_to
    links:
      - "[[../README|Documentation hub]]"
---

# Executable examples

## Runnable scripts and Markdown pages

Two kinds of example live in this repository, and they answer different
questions.

| Where | What it is | Use it to |
|---|---|---|
| [`examples/`](../../examples/README.md) at the repository root | Six scripts you run with `python examples/<name>.py`; each writes figures, tables or RDF | See a complete run and copy it as a starting point |
| This folder | Markdown pages that explain API patterns and agent families | Read how a call fits together before writing your own |

The scripts are tested: `code/tests/test_examples.py` runs every one and checks
the files it writes. Start with [`examples/README.md`](../../examples/README.md)
for the gallery, the command for each script and how to adapt one.

Examples in this repository use the installed `cognitive` and `Things`
packages. The smallest discrete example is in the root `README.md`; the full
configuration, inference, figures, and rendering example is the manuscript in
[`docs/manuscript/README.md`](../manuscript/README.md).

## Commands

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
cognitive-benchmark --repetitions 1
cognitive-build-manuscript --no-pdf --output build/manuscript
```

Examples that need output files must receive a temporary or caller-selected
directory. The package does not assume a repository output tree.
