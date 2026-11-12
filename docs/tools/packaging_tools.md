---
title: Packaging tools
type: tool
status: stable
updated: 2026-10-10
created: 2026-08-02
tags:
  - packaging
  - pyproject
  - tools
semantic_relations:
  - type: documents
    links:
      - "[[../../pyproject.toml|Package configuration]]"
---

# Packaging tools

Packaging is defined entirely in `pyproject.toml` (setuptools backend):

- The `cognitive` package maps to `code/tools/src`; `Things` to
  `code/Things`; `scripts` to `code/scripts`.
- Console entry points are declared in
  [`pyproject.toml`](../../pyproject.toml), rather than duplicated here.

Install for development:

```bash
uv sync --locked --extra dev --python 3.12
```

The CI `tests` job builds with the locked development tools and
`python -m build --no-isolation`, checks metadata with `twine check`, and
installs the wheel in an independent environment using exported locked
requirements. Console smoke checks run outside the checkout so an editable
installation cannot hide a broken wheel.

Do not introduce a separate packaging toolchain; extend `pyproject.toml`
when new modules or entry points are added.
