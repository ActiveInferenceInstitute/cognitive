---
title: Linting tools
type: tool
status: stable
created: 2026-08-02
tags:
  - linting
  - ruff
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../development/README|Development resources]]'
---

# Linting tools

The repository uses `ruff` for linting and formatting (configured in
`pyproject.toml`; project style is ruff only, no black):

```bash
ruff check .
ruff format code/Things code/tools/src code/scripts code/tests
```

`ruff check` runs in CI together with mypy and compileall. See
`docs/development/README.md` for the complete development loop.
