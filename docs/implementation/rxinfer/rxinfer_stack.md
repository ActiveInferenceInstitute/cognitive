---
title: The RxInfer ecosystem
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
      - "[[getting_started]]"
      - "[[model_specification]]"
      - "[[message_passing]]"
      - "[[reactive_programming]]"
      - "[[execution_engine]]"
---

# The RxInfer ecosystem

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

RxInfer is one package in the ReactiveBayes ecosystem, which joins three core packages into one framework for reactive message-passing Bayesian inference:

| Package | Role |
|---|---|
| RxInfer.jl | the user-facing API: `infer`, `@autoupdates`, `@initialization`, the inference engine for streaming data |
| GraphPPL.jl | model and constraints specification: `@model`, `@constraints`, `@meta` |
| ReactiveMP.jl | the message-passing engine: factor nodes, update rules, Bethe free energy |
| Rocket.jl | reactive-programming tools: observables, subjects, subscriptions |

ReactiveMP.jl is the successor of ForneyLab.jl: the ideas are shared but the API differs.

A run goes through three phases: model specification (GraphPPL), inference specification (`infer` arguments such as `constraints`, `initialization`, `iterations`), and inference execution (ReactiveMP). See [[getting_started]]. Earlier versions of this page described a compiler with intermediate representation, optimization passes and a code generator; the official documentation does not describe such a pipeline, so that material was removed (see [[compiler_pipeline]]).

## Official documentation

- Home page: <https://docs.rxinfer.com/stable/>
- RxInfer.jl: <https://github.com/ReactiveBayes/RxInfer.jl>
- ReactiveMP.jl: <https://github.com/ReactiveBayes/ReactiveMP.jl>
- GraphPPL.jl: <https://github.com/ReactiveBayes/GraphPPL.jl>
- Rocket.jl: <https://github.com/ReactiveBayes/Rocket.jl>
