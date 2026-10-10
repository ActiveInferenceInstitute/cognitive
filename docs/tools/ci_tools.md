---
title: CI tools
type: tool
status: stable
updated: 2026-10-10
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

The only CI workflow is `.github/workflows/quality.yml` (name `quality`). It
runs on every push and pull request and has three jobs.

## `test-and-validate`

Runs on Python 3.10 and 3.12:

- `python -m pip install -e ".[dev]"`
- installs `pandoc` and `pandoc-crossref` from pinned release binaries so the
  manuscript HTML render runs (the PDF render skips: the runner has no XeLaTeX)
- `python -m pytest -q -p pytest_cov --cov` (branch coverage against the
  `fail_under` floor in `pyproject.toml`)
- `ruff check .`, `mypy code/tools/src code/Things code/scripts`,
  `python -m compileall -q code`
- `python code/scripts/validate_docs.py --json`
- `python code/scripts/verify_links.py . --json`
- `python -m scripts.build_manuscript --no-render` and `--no-pdf`, followed by
  assertions that the rendered HTML has no unresolved markers
- `cognitive-benchmark --repetitions 1 --output <temp>/benchmarks.json`

## `mutation-gate`

`python -m scripts.mutation_check --json` on Python 3.12; a surviving mutation
fails the job. See [[docs/tools/mutation_testing|Mutation testing]].

## `build-package`

Builds the distribution with `python -m build`, runs `twine check`, installs the
wheel, and runs `--help` on seven of the eight console scripts
(`cognitive-mutation-check` is exercised by the mutation job).

Run the same commands locally; see
[[docs/development/README|Development resources]].
