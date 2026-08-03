---
title: Implementation documentation instructions
type: agents
status: stable
created: 2024-01-01
updated: 2026-08-02
tags:
  - agents
  - implementation
  - guide
  - development
semantic_relations:
  - type: organizes
    links:
      - '[[README|Implementation overview]]'
      - '[[../manuscript/README|Executable manuscript]]'
      - '[[rxinfer/README|RxInfer integration]]'
      - '[[../../code/tools/src/models/active_inference/AGENTS|Active inference models]]'
---

# Implementation documentation instructions

The implementation documentation describes how the package's behavior is
built and exercised, and how the RxInfer integration material is organized.

## Sources of truth

- The executable implementation example is the manuscript in
  `docs/manuscript/`; its builder, `code/scripts/build_manuscript.py`,
  defines the runtime behavior it documents.
- The public package API is documented in `docs/api/api_reference.md`.
- The `rxinfer/` subtree documents a Julia probabilistic-programming
  integration; its `docs/` subfolder is vendored RxInfer documentation and is
  not part of the Python package.

## Rules

- Implementation pages must reference symbols that exist in the package.
- Do not present architecture sketches as available classes.
- Keep the `rxinfer/` vendored documentation tree read-only; record local
  notes in the top-level `rxinfer/*.md` pages instead.
- Update `docs/implementation/README.md` when the implementation surface
  changes.
