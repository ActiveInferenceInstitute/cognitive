---
title: RxInfer documentation instructions
type: agents
status: stable
created: 2025-01-01
updated: 2026-10-10
tags:
  - agents
  - rxinfer
  - implementation
semantic_relations:
  - type: organizes
    links:
      - "[[README|RxInfer overview]]"
      - "[[getting_started|Getting started]]"
      - "[[model_specification|Model specification]]"
      - "[[variational_inference|Variational inference]]"
      - "[[message_passing|Message passing]]"
      - "[[factor_graphs|Factor graphs]]"
---

# RxInfer documentation instructions

This subtree describes RxInfer.jl, an external Julia library. It is not used by the Python package.

## Structure

- The `rxinfer/*.md` pages are local summaries of the official documentation.
- The folder does not hold a copy of the RxInfer documentation source: it goes out of date with each release (RxInfer.jl is at 5.x), so the pages link to <https://docs.rxinfer.com/stable/>.

## Rules

- Every local page begins with a note that it describes an external project and links to <https://docs.rxinfer.com/stable/>.
- State only what the official documentation supports. Do not invent API names, keyword arguments, or internals.
- Do not present Python classes for RxInfer components; there are none.
- Julia snippets are not run here; take them from the official documentation.
- Use only URLs that resolve; the verified ones are listed in each page's "Official documentation" section.
- Keep `README.md` in sync with the pages that exist.
