---
title: CI tools
type: tool
status: stable
updated: 2026-11-12
created: 2026-08-02
tags:
  - ci
  - continuous_integration
  - tools
semantic_relations:
  - type: documents
    links:
      - "[[docs/development/README|Development resources]]"
---

# CI tools

The CI workflow is [quality.yml](../../.github/workflows/quality.yml), named
`quality`, triggered by pushes and pull requests. It defines a consolidated
`tests` job and an independent `gnn-conformance` job. Superseded runs on the same
ref are cancelled by workflow concurrency.

## Shared environment and entry point

Both jobs use
`ActiveInferenceInstitute/.github/.github/actions/setup-python-uv@7159e67315330aac8e414aebae0ad1e999a41448`
and the committed `uv.lock` for the project environment on Python 3.12.
CI calls the same root `./run.sh` entry point as local development, which invokes
`uv run --locked --extra dev --python 3.12 python code/scripts/gate.py ARM`.
The pinned GNN parser retains its independent interpreter; it is not installed
into the project environment.

## `tests`

The consolidated job runs the `lint`, `types`, `test-full`, `docs`, and
`manuscript` arms. These cover Ruff lint and formatting checks, compileall,
mypy, the entire pytest suite with branch coverage, documentation and OKN
checks, and the no-render and HTML manuscript builds. PDF rendering is opt-in
with `./run.sh manuscript --pdf` and requires XeLaTeX.

The job also retains benchmark, captured OKN evidence replay and assertion-path
explanation checks, mutation qualification, and distribution build/install and
console-entry-point checks. These are no longer separate project-install jobs.
Build and package-validation tools are locked development dependencies.
The final replay steps use `uv run --locked --extra dev --extra replay`:
the optional extra locks `daf-okn` and `rdflib==7.6.0` without adding the
replay environment to ordinary local gate invocations.
The default local `./run.sh` instead selects `test`: only the core-model files
listed in [Development resources](../development/README.md), with
`-m 'not visualization'` and no coverage. This is narrower feedback, not an
equivalent substitute for full-suite coverage or workload. The before fast
sample reconstructs the same selection against the original source; there was
no separately named fast CI arm before this entry point.

## `gnn-conformance`

Its display name is **`gnn conformance (pinned)`**. Preserve both this display
name and the `gnn-conformance` job key for check consumers.

The job reads the canonical `gnn-importer-pin/v1` declaration in
[`specs/gnn-pin.json`](../../specs/gnn-pin.json), checks that its tag still peels
to its commit, and provisions the parser in a separate runtime. With
`GNN_RUNTIME_PYTHON` set, `./run.sh gnn` validates the single declaration
`specs/discrete_variational.gnn.md` and runs its conformance tests. Parser file
digests are verified before accepting its verdict. Missing provisioning fails
rather than counting as a pass. See [GNN conformance](../../.github/GNN.md) for
the exact pin and scope.

## Execution evidence

Hosted Actions execution is currently blocked by the organization's billing
lock. This page describes the configured contract, not a successful hosted run.
Local measurements and their provenance live only in the generated
[gate measurements](../development/gate-measurements.md), sourced from
[gate-measurements.json](../development/gate-measurements.json).
