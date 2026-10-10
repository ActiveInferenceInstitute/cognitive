---
title: Reactive programming in RxInfer
type: documentation
status: stable
created: 2024-03-20
updated: 2026-10-10
tags:
  - rxinfer
  - julia
aliases:
  - reactive_programming_concepts
semantic_relations:
  - type: relates
    links:
      - "[[streaming_inference]]"
      - "[[message_passing]]"
      - "[[rxinfer_stack]]"
---

# Reactive programming in RxInfer

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

RxInfer builds on Rocket.jl, a reactive-programming library for Julia. A stream (observable) emits values over time; consumers subscribe to it and are notified when a new value arrives. RxInfer uses this in two places: the message-passing engine recomputes messages as soon as their inputs change, and `infer` can consume a data stream and publish a stream of posterior updates.

```julia
using Rocket

datastream   = RecentSubject(Bool)
observations = labeled(Val((:y, )), combineLatest(datastream))   # emits NamedTuples

subscription = subscribe!(observations, (obs) -> println("got ", obs))
next!(datastream, true)
unsubscribe!(subscription)
```

`labeled` turns the stream into `NamedTuple`s whose keys are model argument names, which is the form `infer` expects. Always unsubscribe from streams that are no longer needed. Using the stream for inference is described in [[streaming_inference]].

## Official documentation

- Reactive programming concept page: <https://docs.rxinfer.com/stable/concepts/reactive-programming/>
- Rocket.jl: <https://github.com/ReactiveBayes/Rocket.jl>
