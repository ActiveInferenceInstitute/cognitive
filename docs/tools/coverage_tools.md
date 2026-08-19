---
title: Coverage tools
type: tool
status: stable
created: 2026-08-02
tags:
  - coverage
  - testing
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../development/README|Development resources]]'
---

# Coverage tools

Coverage runs through `pytest-cov`, configured entirely in `pyproject.toml`:

```bash
python -m pytest -q --cov
```

## What is measured

`[tool.coverage.run]` sets `source` to `code/tools/src`, `code/Things`, and
`code/scripts`, and enables **branch** coverage. Branch measurement matters
here because the package's correctness lives in its validation guards: a
statement-only run counts a `raise ValueError(...)` guard as covered when only
its passing side ever executes, which is precisely the case a test should be
proving.

`[tool.coverage.report]` declares **no `include` filter**. It previously listed
five modules, which reported 93% while the measured figure across every module
in `source` was 70%. A filtered report is not a coverage number for the
package; it is a coverage number for whichever files someone chose to look at.

## The floor

`fail_under = 87` is the whole-package branch figure measured in the leaner of
the two environments — one without Pandoc, where
`test_builder_renders_html_when_the_toolchain_is_present` skips. With
`pandoc` and `pandoc-crossref` on `PATH` the same suite measures roughly a
point and a half higher.

Raise the floor when new tests raise the measurement. Do not raise it by
narrowing what is measured.

## Reading the gaps

The report prints missing lines and partial branches (`show_missing = true`).
An entry such as `136->138` is a branch that was only ever taken one way — the
guard exists, and no test has made it fire.

The CI workflow runs the same command, so a local pass and a CI pass mean the
same thing. Tests write artifacts only to temporary directories.
