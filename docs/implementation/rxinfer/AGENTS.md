---
title: RxInfer integration documentation instructions
type: agents
status: stable
created: 2025-01-01
updated: 2026-08-02
tags:
  - agents
  - rxinfer
  - implementation
  - probabilistic_programming
  - active_inference
semantic_relations:
  - type: organizes
    links:
      - '[[README|RxInfer overview]]'
      - '[[getting_started|Getting started]]'
      - '[[model_specification|Model specification]]'
      - '[[variational_inference|Variational inference]]'
      - '[[message_passing|Message passing]]'
      - '[[factor_graphs|Factor graphs]]'
---

# RxInfer integration documentation instructions

This subtree documents the RxInfer (Julia) probabilistic-programming
integration used for message-passing and variational-inference models.

## Structure

- The top-level `rxinfer/*.md` pages are local integration notes: model
  specification, message passing, variational inference, factor graphs, and
  the systems-engineering guidebook.
- The `docs/` subfolder is vendored RxInfer documentation (a Documenter.jl
  site: `make.jl`, `Project.toml`, `src/`). Treat it as a read-only third-party
  snapshot; edits there should only be made when refreshing the vendor copy.

## Rules

- RxInfer notes are conceptual and use Julia syntax; they are not part of the
  Python package and are excluded from its linters.
- Do not present Python classes for RxInfer components; the integration is
  documented in Julia terms.
- Keep `docs/implementation/rxinfer/AGENTS.md` and `README.md` in sync when
  the vendor copy is refreshed.
