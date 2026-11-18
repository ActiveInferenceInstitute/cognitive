---
title: Contribution guide
type: guide
status: stable
updated: 2026-10-09
---

# Contribution guide

## Setup

Follow the [[docs/repo_docs/getting_started|getting-started guide]] to open a
fresh checkout in the locked Python 3.12 development environment.

## Implementation requirements

Changes to runtime behavior should include a regression test, explicit input
validation, a documented public import path, and an update to the executable
manuscript when the mathematical contract changes. Random behavior must accept
an explicit seed. File outputs must be supplied by the caller or resolved from
the configuration file.

Tests should exercise real numerical paths and inspect actual outputs. Use
`tmp_path` for generated files. Do not leave reports, figures, or build trees
in the repository.

## Required checks

Run the shared gates from the repository root.

```bash
./run.sh lint
./run.sh types
./run.sh test-full
./run.sh docs
```

`test-full` enforces the branch coverage floor in `pyproject.toml`. For faster
feedback, `./run.sh test` runs the selected core-model tests. The
[[docs/development/README|development guide]] describes the local arms and the
additional jobs and steps in the configured quality workflow.

## Knowledge base changes

Regenerate the affected catalogs after adding, renaming or retitling pages.
When a change affects the RDF export, refresh the generated graph documentation
and semantic review before running the docs gate. Follow the
[[docs/development/README#Documentation maintenance|maintenance instructions]]
and the [[docs/okn/README|OKN export guide]].

The quality workflow also replays a capture bound to the N-Quads and join query
hashes. If either changes, prepare a new capture and matching explanation,
preserve the earlier evidence, and update the workflow to replay the new capture.
See the [[docs/okn/evidence/README|capture guide]] and
[[docs/okn/join-support|join explanation guide]].

Keep commits focused and include the observed validation results in the pull
request description.
