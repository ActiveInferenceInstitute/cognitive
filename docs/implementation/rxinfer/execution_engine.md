---
title: Running inference in RxInfer
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
      - "[[streaming_inference]]"
      - "[[free_energy]]"
      - "[[message_passing]]"
---

# Running inference in RxInfer

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

Inference is executed through the `infer` function, in one of two settings.

| Setting | Input | Returns |
|---|---|---|
| Static (batch) | `data = (y = ..., )` | an `InferenceResult` with `posteriors`, optionally `free_energy` and predictions |
| Streaming | `datastream = ...` with `autoupdates` | an `RxInferenceEngine`; start it with `RxInfer.start` and stop it with `RxInfer.stop` |

Documented arguments of `infer` include `model`, `data`, `datastream`, `constraints`, `meta`, `initialization`, `returnvars`, `predictvars`, `historyvars`, `iterations`, `free_energy`, `free_energy_diagnostics`, `options`, `catch_exception`, `callbacks`, `addons`, `postprocess` and (streaming) `autoupdates`, `autostart`, and `keephistory`. See the inference overview for exact semantics.

`callbacks` is the hook for observing the engine: the built-in handlers include `RxInferBenchmarkCallbacks`, `RxInferTraceCallbacks` and `StopEarlyIterationStrategy`.

```julia
result = infer(
    model = coin_model(a = 2.0, b = 7.0),
    data  = (y = [1, 0, 1, 1, 0], ),
    callbacks = (
        after_inference = (event) -> println("Inference completed"),
    )
)
```

The engine itself is ReactiveMP.jl. Earlier versions of this page listed engine classes (message processor, scheduler, memory manager, parallel executor) that are not part of the documented API and were removed.

## Official documentation

- Inference overview: <https://docs.rxinfer.com/stable/manuals/inference/overview/>
- Static inference: <https://docs.rxinfer.com/stable/manuals/inference/static/>
- Streaming inference: <https://docs.rxinfer.com/stable/manuals/inference/streamlined/>
- Callbacks: <https://docs.rxinfer.com/stable/manuals/inference/callbacks/>
- Early stopping: <https://docs.rxinfer.com/stable/manuals/inference/early-stopping/>
