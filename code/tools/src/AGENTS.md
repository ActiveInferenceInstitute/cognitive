---
title: Package source instructions
type: agents
status: stable
updated: 2026-11-03
---

# AGENTS.md — `code/tools/src/`

- This tree is installed as `cognitive` (`package-dir` in `pyproject.toml`). A new
  subpackage must be added to the `[tool.setuptools] packages` list, or it is not
  installed.
- Import public behavior from `cognitive`, not from a filesystem path. Keep
  `py.typed` in the package root.
- Run `ruff check .`, `mypy code/tools/src code/Things code/scripts` and
  `python -m pytest -q --cov`; do not pass `--cov=<path>`.
- Keep return types annotated and public functions documented; the package is
  type-checked in CI.
- Parent rules: [../AGENTS.md](../AGENTS.md).
