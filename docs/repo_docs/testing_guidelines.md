---
title: Testing guidelines
type: guide
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[docs/repo_docs/unit_testing|Unit testing]]"
      - "[[docs/repo_docs/validation|Validation]]"
---

# Testing guidelines

Every change to code in `code/tools/src`, `code/Things`, or `code/scripts` needs
a test in `code/tests/` that fails without the change. Run `python -m pytest -q
--cov` before opening a change; the coverage floor in `pyproject.toml` is part
of the gate. Layout, fixtures, conventions, and an example are in
[[docs/repo_docs/unit_testing|Unit testing]]; the full set of gates is in
[[docs/repo_docs/validation|Validation]].
