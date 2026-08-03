---
title: Coverage tools
type: tool
status: stable
created: 2026-08-02
tags:
  - coverage
  - testing
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../development/README|Development resources]]'
---

# Coverage tools

Test coverage uses `pytest-cov` with a `--cov-fail-under=90` floor declared
in `pyproject.toml`:

```bash
python -m pytest -q --cov=code/tools/src --cov-report=term-missing
```

The CI workflow runs the same coverage gate. Tests write artifacts only to
temporary directories.
