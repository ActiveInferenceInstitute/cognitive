---
title: API documentation tools
type: tool
status: stable
created: 2026-08-02
tags:
  - api
  - documentation
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../api/README|API documentation]]'
      - '[[../api/api_reference|API reference]]'
---

# API documentation tools

The canonical API documentation is `docs/api/`:

- `docs/api/api_reference.md` — the reference for the installed `cognitive`
  and `Things` packages.
- `docs/api/README.md` — documentation policy: describe only symbols
  importable from the installed packages.

The documentation validator checks that public exports exist:

```bash
python code/scripts/validate_docs.py --json
```

When a public symbol changes, update `docs/api/api_reference.md` in the same
change and re-run the validator.
