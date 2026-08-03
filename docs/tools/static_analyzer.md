---
title: Static analysis
type: tool
status: stable
created: 2026-08-02
tags:
  - static_analysis
  - mypy
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../development/README|Development resources]]'
---

# Static analysis

Static type checking uses `mypy` (configured in `pyproject.toml`):

```bash
mypy code/tools/src code/Things code/scripts
```

`python -m compileall -q code` is the syntax gate. Both run in CI and are
part of the local development loop (see `docs/development/README.md`).
