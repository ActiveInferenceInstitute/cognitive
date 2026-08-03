---
title: CI tools
type: tool
status: stable
created: 2026-08-02
tags:
  - ci
  - continuous_integration
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../development/README|Development resources]]'
---

# CI tools

The repository's continuous integration is defined in
`.github/workflows/quality.yml`. It runs on push and pull request for Python
3.10 and 3.12:

- `python -m pip install -e ".[dev]"`
- pytest under coverage (`coverage run -m pytest -q`, `coverage report -m`)
- `ruff check .`
- `mypy code/tools/src code/Things code/scripts`
- `python -m compileall -q code`
- `python code/scripts/validate_docs.py --json`
- `python code/scripts/verify_links.py . --json`

Run the same commands locally; see `docs/development/README.md`.
