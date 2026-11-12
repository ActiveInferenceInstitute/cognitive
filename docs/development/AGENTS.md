---
title: Development documentation instructions
type: agents
status: stable
created: 2024-01-01
updated: 2026-10-09
tags:
  - development
  - agents
  - workflow
  - collaboration
semantic_relations:
  - type: supports
    links:
      - "[[README|Development resources]]"
      - "[[../api/README|API documentation]]"
      - "[[../implementation/README|Implementation]]"
---

# Development documentation instructions

The development documentation records how to work on this repository: the
development loop, quality gates, and conventions. It does not describe
automated development agents.

## Development loop

Use the shared root entry point rather than copying gate subprocess commands:

```bash
./run.sh                 # defaults to test
./run.sh lint
./run.sh types
./run.sh test-full
./run.sh docs
./run.sh gnn             # requires provisioned GNN_RUNTIME_PYTHON
./run.sh manuscript      # no-render + HTML; --pdf adds PDF
```

It invokes
`uv run --locked --extra dev --python 3.12 python code/scripts/gate.py ARM`.
The default `test` arm selects only the core-model files documented in
`docs/development/README.md`, with `-m 'not visualization'` and no coverage.
It provides narrower feedback, not equivalent full-suite coverage or workload.
`test-full` runs the entire suite with branch coverage. GNN provisioning is
explicit through `./run.sh gnn --provision`, which permits network access.

The complete contract, prerequisites, independent benchmark and mutation
commands, and consolidated CI job descriptions are in
`docs/development/README.md`. Hosted Actions execution is currently blocked
by the organization's billing lock; do not describe configuration or local
evidence as a hosted pass. Measured values belong only in the generated
`gate-measurements.md` and its source `gate-measurements.json`.

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
