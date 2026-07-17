---
title: Executable examples
type: documentation
status: stable
---

# Executable examples

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
