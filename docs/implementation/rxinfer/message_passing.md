---
title: Message passing in RxInfer
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
      - "[[factor_graphs]]"
      - "[[variational_inference]]"
      - "[[free_energy]]"
      - "[[reactive_programming]]"
---

# Message passing in RxInfer

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

RxInfer's core algorithms are belief propagation and variational message passing. Belief propagation gives exact marginals where it applies; variational message passing is an approximation that covers a larger range of models.

The inference engine does not know which algorithm it is running. It exchanges messages between nodes, and the choice of algorithm comes from the factorization constraints given with `@constraints` around each factor node: different constraints select different update rules (see [[variational_inference]]). The engine can run different algorithms in different parts of one factor graph. Message passing is reactive and schedule-free: a message is recomputed when its inputs change.

Custom nodes and update rules are added with RxInfer's rule and node macros; the official pages on custom nodes and on inference without explicit rules cover this. The invented scheduling, caching and parallelism options that earlier versions of this page listed (`MaxProduct()`, `cache_messages`, `parallel`) are not documented RxInfer options and were removed.

## Official documentation

- Message passing concept page: <https://docs.rxinfer.com/stable/concepts/message-passing/>
- Inference overview: <https://docs.rxinfer.com/stable/manuals/inference/overview/>
- Custom nodes and rules: <https://docs.rxinfer.com/stable/manuals/inference/create-node/>
- Inference without defining rules: <https://docs.rxinfer.com/stable/manuals/inference/undefinedrules/>
- ReactiveMP.jl: <https://github.com/ReactiveBayes/ReactiveMP.jl>
