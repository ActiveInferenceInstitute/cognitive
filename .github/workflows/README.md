---
title: CI workflows
type: documentation
status: stable
updated: 2026-11-12
---

# `.github/workflows/`

[quality.yml](quality.yml) is configured for pushes and pull requests.

| Job key | Display name / scope |
|---|---|
| `tests` | Consolidated lint, types, full-suite branch coverage, docs and OKN checks, manuscript no-render and HTML, benchmarks, evidence replay, mutation qualification, and package checks. |
| `gnn-conformance` | `gnn conformance (pinned)`: the single `specs/discrete_variational.gnn.md` declaration against the parser pinned by `specs/gnn-pin.json`, in an independent interpreter. |

Both jobs use
`ActiveInferenceInstitute/.github/.github/actions/setup-python-uv@7159e67315330aac8e414aebae0ad1e999a41448`
and `uv.lock` for their Python 3.12 project environments. Shared gates run through
root `./run.sh`, which invokes
`uv run --locked --extra dev --python 3.12 python code/scripts/gate.py ARM`.
The GNN parser environment stays separate from the project environment.

`concurrency` cancels a superseded run on the same ref. Preserve the
`gnn-conformance` key and `gnn conformance (pinned)` display name.
See [CI tools](../../docs/tools/ci_tools.md) and [GNN conformance](../GNN.md).

Hosted Actions execution is currently blocked by the organization's billing
lock; these job definitions are not a claim of hosted success. Local evidence
is recorded in [gate measurements](../../docs/development/gate-measurements.md).
