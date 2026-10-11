---
title: Implementation documentation index
type: documentation
status: stable
updated: 2026-10-10
semantic_relations:
  - type: organizes
    links:
      - "[[AGENTS|Implementation documentation instructions]]"
      - "[[implementation_guides|Implementation guides]]"
      - "[[implementation_guides_index|Implementation guides index]]"
      - "[[implementation_patterns|Implementation patterns]]"
      - "[[rxinfer/README|RxInfer integration]]"
      - "[[../manuscript/README|Executable manuscript]]"
      - "[[../api/README|API documentation]]"
---

# Implementation documentation index

This directory documents how to implement agents with the `cognitive` package
and holds notes on the external Julia library RxInfer.

## Pages

- [[implementation_guides]] and [[implementation_guides_index]]: the guides that
  exist and where to find them.
- [[implementation_patterns]]: patterns the package follows, with runnable
  snippets.

## RxInfer

`rxinfer/` describes RxInfer.jl, a Julia library for reactive message-passing
inference. It is an external project: the Python package does not call it. The
folder holds local explanatory notes that link to the official documentation. See
[[rxinfer/README]].

## Sources of truth

- The runnable reference implementation is the manuscript build
  (`cognitive-build-manuscript`, [[../manuscript/README|manuscript]]).
- The public API is in [[../api/README|docs/api]].
- The package source is `code/tools/src` (import name `cognitive`) and
  `code/Things`.
