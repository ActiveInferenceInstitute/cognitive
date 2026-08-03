---
title: Dependency analysis
type: tool
status: stable
created: 2026-08-02
tags:
  - dependencies
  - pyproject
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../../pyproject.toml|Package configuration]]'
---

# Dependency analysis

The single source of truth for dependencies is `pyproject.toml`:

- `[project].dependencies` — runtime dependencies (click, jinja2,
  matplotlib, networkx, numpy, Pillow, plotly, PyYAML, seaborn).
- `[project].optional-dependencies].dev` — development tools (coverage,
  mypy, pytest, pytest-cov, ruff, types-PyYAML).

Add or remove dependencies only by editing `pyproject.toml`; do not add a
separate `requirements.txt` unless a Thing folder documents a local need.
