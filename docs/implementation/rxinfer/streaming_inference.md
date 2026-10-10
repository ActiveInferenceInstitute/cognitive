---
title: Streaming inference in RxInfer
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
      - "[[reactive_programming]]"
      - "[[model_specification]]"
      - "[[message_passing]]"
---

# Streaming inference in RxInfer

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

In streaming (online) inference, beliefs are updated each time an observation arrives. The model takes the prior parameters as arguments, `@autoupdates` says how to refresh them from the new posterior, and `infer` is called with `datastream` instead of `data`.

```julia
using RxInfer, Rocket

@model function beta_bernoulli_online(y, a, b)
    θ ~ Beta(a, b)
    y ~ Bernoulli(θ)
end

autoupdates = @autoupdates begin
    a, b = params(q(θ))
end

datastream   = RecentSubject(Bool)
observations = labeled(Val((:y, )), combineLatest(datastream))

engine = infer(
    model          = beta_bernoulli_online(),
    datastream     = observations,
    autoupdates    = autoupdates,
    returnvars     = (:θ, ),
    initialization = @initialization(q(θ) = Beta(1, 1)),
    autostart      = false
)

subscription = subscribe!(engine.posteriors[:θ], (q) -> println("posterior: ", q))
RxInfer.start(engine)
next!(datastream, true)
RxInfer.stop(engine)
```

Points from the official manual:

- `data` and `datastream` are mutually exclusive; one is required.
- `@autoupdates` needs an `initialization` for the latent states it reads.
- With `autostart = false` the engine subscribes to the stream only after `RxInfer.start`.
- `returnvars` is a tuple of symbols in the streaming setting.
- `keephistory` and `historyvars` keep past posteriors.
- Static datasets also work with `autoupdates`, which is useful for testing.

## Official documentation

- Streaming inference: <https://docs.rxinfer.com/stable/manuals/inference/streamlined/>
- Autoupdates: <https://docs.rxinfer.com/stable/manuals/inference/autoupdates/>
- Static versus streaming: <https://docs.rxinfer.com/stable/manuals/inference/static-vs-streamlined/>
