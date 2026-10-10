---
title: Variational inference in RxInfer
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
      - "[[rxinfer_variational_inference]]"
      - "[[free_energy]]"
      - "[[message_passing]]"
      - "[[factor_graphs]]"
---

# Variational inference in RxInfer

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

RxInfer performs variational inference by message passing. Given a model `p(s, y)` and data `y`, it looks for the distribution `q(s)` inside a chosen family `Q` that minimizes the variational free energy,

```text
q* = argmin over q in Q of  E_q(s)[ log q(s) - log p(s, y) ]
```

With no extra constraints the optimum is the exact posterior. Constraints shrink `Q` to trade accuracy for tractability; they are written with `@constraints` ([[rxinfer_variational_inference]]). The objective that the engine optimizes is the Bethe free energy ([[free_energy]]).

## Two kinds of constraint

- Factorization constraints, such as `q(μ, τ) = q(μ)q(τ)` (mean field).
- Functional form constraints on marginals (`q(x) :: ...`) or messages (`μ(x) :: ...`), for example `q(x) :: PointMassFormConstraint()`.

Some constraints make the problem iterative, so the engine needs initial marginals or messages (`@initialization`) and a number of `iterations`.

## Not covered here

Earlier versions of this page listed natural-gradient, stochastic-VI, amortized-VI and `ELBO` options for `infer`. Those keyword arguments are not part of the documented `infer` interface and were removed. Non-conjugate models, where analytical update rules do not apply, have their own section in the official manual.

## Official documentation

- Variational inference concept page: <https://docs.rxinfer.com/stable/concepts/variational-inference/>
- Constraints specification: <https://docs.rxinfer.com/stable/manuals/variational/constraints-specification/>
- Functional forms: <https://docs.rxinfer.com/stable/manuals/variational/functional-forms/>
- Non-conjugate inference: <https://docs.rxinfer.com/stable/manuals/inference/nonconjugate/>
