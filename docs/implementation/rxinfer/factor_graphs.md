---
title: Factor graphs in RxInfer
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
      - "[[model_specification]]"
      - "[[message_passing]]"
      - "[[variational_inference]]"
---

# Factor graphs in RxInfer

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

A model is a function that factorizes over smaller factors,

```text
f(y, x) = product over a of f_a(y_a, x_a)
```

where `y` are observed variables and `x` are latent variables. A factor graph draws each factor as a node and each variable as an edge or node connecting the factors it appears in. RxInfer builds a Forney-style factor graph from the `@model` specification and runs message passing on it.

## Inspecting a graph

With `Cairo` and `GraphPlot` installed (they are not dependencies of RxInfer), the graph of a conditioned model can be drawn and queried:

```julia
using RxInfer, Cairo, GraphPlot

@model function coin_model(y, a, b)
    θ  ~ Beta(a, b)
    y .~ Bernoulli(θ)
end

conditioned = coin_model(a = 2.0, b = 7.0) | (y = [true, false, true], )
model = RxInfer.create_model(conditioned)

GraphPlot.gplot(RxInfer.getmodel(model))
RxInfer.getrandomvars(model)
RxInfer.getdatavars(model)
RxInfer.getconstantvars(model)
RxInfer.getfactornodes(model)
```

## Official documentation

- Factor graphs concept page: <https://docs.rxinfer.com/stable/concepts/factor-graphs/>
- Getting started (graph inspection): <https://docs.rxinfer.com/stable/manuals/getting-started/>
