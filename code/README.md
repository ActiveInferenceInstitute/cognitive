---
title: Repository code
type: documentation
status: stable
updated: 2026-11-03
---

# Repository code

Four trees, mapped onto three installed packages by `[tool.setuptools]` in
`pyproject.toml`:

| Directory | Installed as | Contents |
|---|---|---|
| [`Things/`](Things/README.md) | `Things` | `Simple_POMDP` and `Continuous_Generic` agents, plus design-only folders |
| [`tools/src/`](tools/src/README.md) | `cognitive` | the framework, matrix operations, visualization and CLI |
| [`scripts/`](scripts/README.md) | `scripts` | documentation, graph, manuscript and verification commands |
| [`tests/`](tests/README.md) | not installed | the pytest suite (`testpaths = ["code/tests"]`) |

`code/` is not itself a package; imports use `cognitive`, `Things` or `scripts`
after an editable install (`pip install -e ".[dev]"`).

## Checks

```bash
python -m pytest -q --cov          # branch coverage, floor in pyproject.toml
ruff check .
mypy code/tools/src code/Things code/scripts
python code/scripts/validate_docs.py --json
```

The same commands run in `.github/workflows/quality.yml`.
