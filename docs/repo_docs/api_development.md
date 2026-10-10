---
title: API development
type: guide
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[docs/api/api_reference|API reference]]"
      - "[[docs/repo_docs/model_implementation|Model implementation]]"
      - "[[docs/repo_docs/unit_testing|Unit testing]]"
---

# API development

The public API of the `cognitive` package is whatever `cognitive.__all__`
exports; [[docs/api/api_reference|the API reference]] documents it and its
examples are executed against the tree. When you change a public symbol:

- Add type hints and a docstring (the code style is `snake_case`, ruff with a
  100-column line length, and mypy over `code/tools/src`, `code/Things`, and
  `code/scripts`).
- Validate inputs at the boundary and raise `ValueError` with a message that
  names the offending parameter, as the existing classes do.
- Add a test in `code/tests/` (see [[docs/repo_docs/unit_testing|Unit testing]])
  and update the reference page and `CHANGELOG.md`. A change to a computed
  result is marked **behaviour change** there.
- Run `python code/scripts/validate_docs.py --json`, which fails if a name in
  `cognitive.__all__` cannot be resolved on the package.
