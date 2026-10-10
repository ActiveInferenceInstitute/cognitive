---
title: A working checklist for RxInfer models
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
      - "[[rxinfer_variational_inference]]"
      - "[[streaming_inference]]"
      - "[[execution_engine]]"
---

# A working checklist for RxInfer models

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

A short workflow assembled from features that the official documentation describes. It is a suggestion, not an official procedure.

1. **Write the model** with `@model` ([[model_specification]]). Start with conjugate pairs (Beta-Bernoulli, Normal-Gamma), for which RxInfer has analytical update rules.
2. **Check it on simulated data** whose true parameters you know, as the getting-started tutorial does with a coin of known bias.
3. **Add constraints only when needed** ([[rxinfer_variational_inference]]), and add an `initialization` and `iterations` if the problem becomes iterative.
4. **Watch convergence** with `free_energy = true` and look for a converged value rather than a monotone decrease ([[free_energy]]).
5. **Debug** with the official tools: the memory addon (a trace of the computations behind each message and marginal), `RxInferTraceCallbacks`, and the sharp-bits pages for the common rule-not-found and stack-overflow errors.
6. **Move to streaming** with `datastream` and `@autoupdates` once the static model works ([[streaming_inference]]).
7. **Use early stopping and benchmark callbacks** to bound iterations and measure runtime ([[execution_engine]]).

Earlier versions of this guide defined requirement, architecture, deployment and operations template structures as if they were RxInfer features; they are not, and were removed. See [[advanced_systems_engineering_guidebook]].

## Official documentation

- Debugging: <https://docs.rxinfer.com/stable/manuals/debugging/>
- Sharp bits: <https://docs.rxinfer.com/stable/manuals/sharpbits/overview/>
- Performance tips: <https://docs.rxinfer.com/stable/manuals/performance-tips/>
- FAQ: <https://docs.rxinfer.com/stable/manuals/faq/>
