---
title: Model macro internals
type: documentation
status: stable
created: 2024-03-20
updated: 2026-10-10
tags:
  - rxinfer
  - julia
semantic_relations:
  - type: relates
    links:
      - "[[model_macro_paradigm]]"
      - "[[model_specification]]"
      - "[[compiler_pipeline]]"
---

# Model macro internals

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

This page used to describe an abstract-syntax-tree framework for RxInfer, with type-theory and category-theory sections and invented AST node types. None of that is documented by RxInfer or GraphPPL, and it was removed.

What can be said from the official documentation: `@model` is a macro from GraphPPL.jl that takes a Julia function and builds a factor graph from it when the model is conditioned on data ([[model_specification]]). How GraphPPL parses the function body is described in the GraphPPL.jl repository.

Recommendation: keep only as a pointer, or merge into [[model_macro_paradigm]].

## Official documentation

- GraphPPL.jl: <https://github.com/ReactiveBayes/GraphPPL.jl>
