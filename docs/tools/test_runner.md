---
title: Test runner
type: tool
status: stable
created: 2026-08-02
tags:
  - testing
  - pytest
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../../code/tests/README|Test suite]]'
---

# Test runner

The test suite lives in `code/tests/` and runs with pytest:

```bash
python -m pytest -q
python -m pytest -q --cov=code/tools/src --cov-report=term-missing
```

`code/tests/run_tests.py` is a convenience runner. Tests use real data and
computation (no fixture-only assertions) and write artifacts only to
temporary directories. See `docs/repo_docs/unit_testing.md` for
conventions.
