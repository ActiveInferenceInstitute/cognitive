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
- Console entry points: `cognitive-create-node`, `cognitive-verify-links`,
  `cognitive-validate-docs`, `cognitive-normalize-frontmatter`,
  `cognitive-kb-index`, `cognitive-benchmark`, `cognitive-build-manuscript`,
  `cognitive-mutation-check`.

Install for development:

```bash
python -m pip install -e ".[dev]"
```

The CI `build-package` job builds the distribution with `python -m build`,
checks it with `twine check`, and installs the wheel.

Do not introduce a separate packaging toolchain; extend `pyproject.toml`
when new modules or entry points are added.
