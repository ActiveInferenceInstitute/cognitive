---
title: Examples documentation instructions
type: agents
status: stable
updated: 2026-10-13
semantic_relations:
  - type: organizes
    links:
      - "[[README|Examples overview]]"
      - "[[../manuscript/README|Executable manuscript]]"
---

# Examples documentation instructions

## Scripts versus pages

The runnable scripts are in `examples/` at the repository root, documented by
[`../../examples/README.md`](../../examples/README.md) and
[`../../examples/AGENTS.md`](../../examples/AGENTS.md). This folder holds the
Markdown pages. Do not copy a script's code into a page: link to the script
instead, so there is one version to keep correct. When a page describes what a
script does, run the script and quote what it prints.

Examples are documentation of behavior that exists in this repository. Prefer
small examples that can be copied into a Python session and validated against the
public `cognitive` or `Things` packages.

## Sources of truth

- The root `README.md` contains the minimal discrete dispatcher example.
- `docs/manuscript/` contains the end-to-end build, runtime, figure, and
  reproducibility example.
- `code/tests/` contains executable behavior checks.
- `code/Things/` contains self-contained implementations with their own local
  documentation.

Do not present fictional classes, commands, package names, or output paths as
working examples. If an example is theoretical, label it as pseudocode and keep
it out of executable Python fences.

## Validation

```bash
python code/scripts/validate_docs.py --json
python -m pytest -q
```

Examples that write artifacts should accept a caller-selected or temporary output
directory and must not assume an untracked repository output tree.
