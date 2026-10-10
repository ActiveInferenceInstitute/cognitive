---
title: The @model macro
type: documentation
status: stable
created: 2024-03-20
updated: 2026-10-10
tags:
  - rxinfer
  - julia
aliases:
  - model_macro
semantic_relations:
  - type: relates
    links:
      - "[[model_specification]]"
      - "[[factor_graphs]]"
      - "[[rxinfer_stack]]"
---

# The @model macro

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

`@model` is exported through RxInfer but implemented in GraphPPL.jl. A model written with it describes a joint distribution as a product of factors; calling the model function with keyword arguments returns a generator, and the graph is built when the generator is conditioned on data (see [[model_specification]]).

The macro's internals (how the function body is parsed, how nodes are created) are documented in the GraphPPL.jl repository, not here. Earlier versions of this page showed invented expansion code and macros such as `@param` and `@reactive`; those are not part of RxInfer's documented API and were removed.

The factor-graph view of a model is described in [[factor_graphs]]. This page is a short pointer and could be merged into [[model_specification]].

## Official documentation

- Model specification: <https://docs.rxinfer.com/stable/manuals/model-specification/>
- GraphPPL.jl: <https://github.com/ReactiveBayes/GraphPPL.jl>
