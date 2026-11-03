---
title: Code instructions
type: agents
status: stable
updated: 2026-11-03
---

# AGENTS.md — `code/`

- Run `python -m pytest -q --cov`, `ruff check .`,
  `mypy code/tools/src code/Things code/scripts` and
  `python code/scripts/validate_docs.py --json` before publishing.
- Do not pass `--cov=<path>`: the configured `source` list and the 90% floor in
  `pyproject.toml` are the whole-package measurement.
- Tests use real arrays and real files; no fixtures that imitate model behavior.
  Write generated artifacts under `tmp_path` or the session output fixtures.
- `code/` is not a package. New importable code belongs under `tools/src/`
  (as `cognitive`), `Things/` (as `Things`) or `scripts/` (as `scripts`), and the
  `[tool.setuptools]` list must name the new package.
- The documentation gate rejects the words in
  [`docs/policy/documentation_terms.yaml`](../docs/policy/documentation_terms.yaml)
  in every tracked text file.
- Parent rules: [../AGENTS.md](../AGENTS.md).
