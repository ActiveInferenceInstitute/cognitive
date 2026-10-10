---
title: Implementation documentation instructions
type: agents
status: stable
created: 2024-01-01
updated: 2026-10-10
tags:
  - agents
  - implementation
  - guide
semantic_relations:
  - type: organizes
    links:
      - "[[README|Implementation overview]]"
      - "[[../manuscript/README|Executable manuscript]]"
      - "[[rxinfer/README|RxInfer notes]]"
      - "[[../../code/tools/src/models/active_inference/AGENTS|Active inference models]]"
---

# Implementation documentation instructions

The implementation documentation describes how to build agents with the package
and holds notes on the external Julia library RxInfer.

## Sources of truth

- The manuscript in `docs/manuscript/`, built by `code/scripts/build_manuscript.py`
  (`cognitive-build-manuscript`), is the runnable end-to-end example.
- The public package API is documented in `docs/api/api_reference.md`.
- The `rxinfer/` subtree describes RxInfer.jl, an external Julia project that the
  Python package does not use. Its `docs/` subfolder is a copy of the RxInfer
  documentation and is not part of the Python package.

## Rules

- Implementation pages must reference symbols that exist in the package.
- Do not present architecture sketches as available classes; mark proposals with
  `status: proposal`.
- Python snippets that use the package must run with the project's interpreter.
- Keep the `rxinfer/docs/` copy unchanged; record local notes in the top-level
  `rxinfer/*.md` pages, and say on each that it describes an external project.
- Update `docs/implementation/README.md` when the pages here change.
