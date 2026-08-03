---
title: Implementation documentation index
type: documentation
status: stable
semantic_relations:
  - type: organizes
    links:
      - '[[AGENTS|Implementation documentation instructions]]'
      - '[[implementation_guides|Implementation guides]]'
      - '[[implementation_patterns|Implementation patterns]]'
      - '[[rxinfer/README|RxInfer integration]]'
      - '[[../manuscript/README|Executable manuscript]]'
      - '[[../api/README|API documentation]]'
---

# Implementation documentation index

This directory documents how the package's behavior is implemented and
exercised.

## Pages

- `implementation_guides.md` and `implementation_guides_index.md` —
  implementation guidance.
- `implementation_patterns.md` — reusable implementation patterns.

## RxInfer integration

`rxinfer/` documents the RxInfer (Julia) probabilistic-programming
integration: local notes on model specification, message passing,
variational inference, factor graphs, streaming inference, and a systems
engineering guidebook, plus a vendored copy of the RxInfer documentation
tree (`rxinfer/docs/`, read-only).

## Sources of truth

- The executable implementation example is the manuscript
  (`../manuscript/README.md`).
- The public API surface is in `../api/`.
- The package source is `code/tools/src` (import name `cognitive`).
