---
title: Model specification in RxInfer
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
      - "[[model_macro_paradigm]]"
      - "[[factor_graphs]]"
      - "[[variational_inference]]"
---

# Model specification in RxInfer

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

Models are written with the `@model` macro, which comes from GraphPPL.jl. It accepts a regular Julia function and builds a Forney-style factor graph from it.

## Syntax

`~` declares a random variable together with its factor node; `.~` is the broadcast form; `:=` declares a deterministic relation.

```julia
@model function coin_toss(y)
    t ~ Beta(1, 1)
    for i in eachindex(y)
        y[i] ~ Bernoulli(t)
    end
end

@model function deterministic_example()
    t ~ Normal(mean = 0.0, variance = 1.0)
    x := exp(t)
    y ~ Normal(mean = x, variance = 1.0)
end
```

Plain `=` cannot be used for a latent variable, because latent variables have no value when the model is created.

## Submodels

A model can use another model as a factor, passing arguments by keyword:

```julia
@model function submodel(z, x)
    t := x + 1
    z ~ Normal(mean = t, var = 1.0)
end

@model function mymodel(y)
    x ~ Normal(mean = 0.0, var = 1.0)
    z ~ submodel(x = x)
    y ~ Normal(mean = z, var = 1.0)
end
```

## Conditioning on data

Model arguments are passed as keyword arguments. The `|` operator conditions the resulting model generator on data, and `RxInfer.create_model` builds the graph:

```julia
conditioned = coin_model(a = 2.0, b = 7.0) | (y = [true, false, true], )
model       = RxInfer.create_model(conditioned)
```

Data that is not known when the model is created is declared with `RxInfer.DeferredDataHandler()`. In practice `infer(model = ..., data = ...)` does the conditioning for you.

## Official documentation

- Model specification: <https://docs.rxinfer.com/stable/manuals/model-specification/>
- GraphPPL.jl: <https://github.com/ReactiveBayes/GraphPPL.jl>
