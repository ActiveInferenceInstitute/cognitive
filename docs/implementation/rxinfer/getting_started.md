---
title: Getting started with RxInfer
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
      - "[[variational_inference]]"
      - "[[message_passing]]"
      - "[[factor_graphs]]"
---

# Getting started with RxInfer

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

RxInfer.jl is a Julia package for Bayesian inference on factor graphs by message passing. It is built from three packages of the ReactiveBayes organisation: ReactiveMP.jl (the message-passing engine), GraphPPL.jl (model and constraints specification) and Rocket.jl (reactive programming). See [[rxinfer_stack]].

## Install and import

```julia
import Pkg
Pkg.add("RxInfer")
using RxInfer
```

## A first model

The official tutorial infers the bias of a coin with a Beta-Bernoulli model. Inference has three steps: specify the model, specify the inference (here, the data), and execute it.

```julia
using RxInfer, Distributions, Random

rng     = MersenneTwister(42)
dataset = rand(rng, Bernoulli(0.75), 10)

@model function coin_model(y, a, b)
    θ  ~ Beta(a, b)
    y .~ Bernoulli(θ)
end

result = infer(
    model = coin_model(a = 2.0, b = 7.0),
    data  = (y = dataset, )
)

result.posteriors[:θ]   # a Beta distribution
```

`infer` conditions the model on the data itself. Missing observations are allowed (`y = [true, false, missing]`); a dataset made only of `missing` returns the prior.

## Where to go next

- [[model_specification]]: the `@model` syntax and conditioning.
- [[variational_inference]] and [[rxinfer_variational_inference]]: constraints and initialization.
- [[streaming_inference]]: data that arrives over time.

## Official documentation

- Getting started: <https://docs.rxinfer.com/stable/manuals/getting-started/>
- Examples: <https://examples.rxinfer.com/>
- Repository: <https://github.com/ReactiveBayes/RxInfer.jl>
- Reference paper: <https://doi.org/10.21105/joss.05161>
