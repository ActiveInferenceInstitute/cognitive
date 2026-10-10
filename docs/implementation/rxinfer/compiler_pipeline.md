---
title: Model-to-inference pipeline
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
      - "[[rxinfer_stack]]"
      - "[[ast_processing]]"
      - "[[execution_engine]]"
---

# Model-to-inference pipeline

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

This page used to describe a compiler with a frontend, an intermediate representation, optimization passes and code generation. The official documentation describes no such pipeline, so the material was removed.

The documented flow is: a `@model` function (GraphPPL) is conditioned on data and turned into a factor graph; `infer` combines the graph with constraints, meta information and initialization; the ReactiveMP engine runs message passing on the graph. See [[rxinfer_stack]] and [[execution_engine]].

Recommendation: this page duplicates [[rxinfer_stack]] and can be merged into it.

## Official documentation

- Message passing: <https://docs.rxinfer.com/stable/concepts/message-passing/>
- Meta specification: <https://docs.rxinfer.com/stable/manuals/inference/meta-specification/>
