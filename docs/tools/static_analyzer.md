---
title: Static analysis
type: tool
status: stable
updated: 2026-10-10
created: 2026-08-02
tags:
  - static_analysis
  - mypy
  - tools
semantic_relations:
  - type: documents
    links:
      - "[[../development/README|Development resources]]"
---

# Static analysis

There is no custom static analyzer; the repository uses these standard tools.

Static type checking uses `mypy` (configured in `pyproject.toml` with
`python_version = "3.12"`):

```bash
mypy code/tools/src code/Things code/scripts
```

`python -m compileall -q code` is the syntax gate. Both run in CI and are
part of the local development loop (see `docs/development/README.md`).
