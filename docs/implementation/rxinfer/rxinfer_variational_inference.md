---
title: Constraints and initialization in practice
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
      - "[[variational_inference]]"
      - "[[free_energy]]"
      - "[[message_passing]]"
---

# Constraints and initialization in practice

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

A worked case from the official constraints manual: a Normal model with unknown mean and precision, under a mean-field assumption.

```julia
using RxInfer

@model function iid_normal(y)
    μ  ~ Normal(mean = 0.0, variance = 1.0)
    τ  ~ Gamma(shape = 1.0, rate = 1.0)
    y .~ Normal(mean = μ, precision = τ)
end

constraints = @constraints begin
    q(μ, τ) = q(μ)q(τ)
end

# With this constraint the problem is iterative, so one marginal needs an initial value.
init = @initialization begin
    q(τ) = vague(Gamma)
end

result = infer(
    model          = iid_normal(),
    data           = (y = rand(NormalMeanPrecision(3.1415, 2.7182), 1000), ),
    constraints    = constraints,
    initialization = init,
    iterations     = 25
)

mean(result.posteriors[:μ][end])
```

The documentation notes that mean-field estimates deviate slightly from the true values, a known property of that factorization.

## Syntax notes

- `@constraints` and `@initialization` accept either a `begin ... end` block or a function definition.
- `q(...)` refers to a marginal and `μ(...)` to a message.
- `MeanField` and other pre-specified constraints can be passed to `infer` directly.
- Convergence can be monitored with the `free_energy` flag of `infer` ([[free_energy]]).

## Official documentation

- Constraints specification: <https://docs.rxinfer.com/stable/manuals/variational/constraints-specification/>
- Initialization: <https://docs.rxinfer.com/stable/manuals/inference/initialization/>
