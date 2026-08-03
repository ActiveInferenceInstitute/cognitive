---
title: Development documentation instructions
type: agents
status: stable
created: 2024-01-01
updated: 2026-08-02
tags:
  - development
  - agents
  - workflow
  - collaboration
semantic_relations:
  - type: supports
    links:
      - '[[README|Development resources]]'
      - '[[../api/README|API documentation]]'
      - '[[../implementation/README|Implementation]]'
---

# Development documentation instructions

The development documentation records how to work on this repository: the
development loop, quality gates, and conventions. It does not describe
automated development agents.

## Development loop

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check .
mypy code/tools/src code/Things code/scripts
python -m compileall -q code
python code/scripts/validate_docs.py --json
python code/scripts/verify_links.py . --json
```

## Conventions

- Python: snake_case, type hints, docstrings on public methods, ruff for lint
  and format (project style, no black).
- Documentation: every content folder has an `AGENTS.md` (instructions) and a
  `README.md` (overview/navigation); YAML frontmatter is required on Markdown
  files; internal cross-references use `[[wiki links]]`.
- Tests: real data and computation, no fixture-only assertions; tests write
  artifacts only to temporary directories.

## Rules

- Update `docs/development/README.md` when the tooling or gate commands
  change.
- Do not document fictional development workflows or agent frameworks; the
  repository's actual tooling lives in `code/scripts/` and the installed CLI
  entry points.
