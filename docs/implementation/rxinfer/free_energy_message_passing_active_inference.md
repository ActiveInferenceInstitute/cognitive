---
title: Active inference with RxInfer
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
      - "[[free_energy]]"
      - "[[message_passing]]"
      - "[[active_inference_examples]]"
      - "[[knowledge_base/cognitive/active_inference]]"
---

# Active inference with RxInfer

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

The Bethe free energy that RxInfer optimizes is a variational free energy over a factor graph. Active inference additionally scores policies by expected free energy. RxInfer does not ship an active inference agent class; agents are written as ordinary `@model` functions, and the official examples site shows how.

The example "Expected free energy minimization via message passing" reproduces the method of the paper "A Message Passing Realization of Expected Free Energy Minimization". It casts expected free energy minimization as variational free energy minimization on a factor graph with epistemic priors, and defines two custom nodes, `Exploration` and `Ambiguity`, with `@node` and `@rule` to supply those priors. It runs an agent in a stochastic maze using `infer` with `@initialization` and `iterations`.

## Relation to this repository

The Python package computes expected free energy for discrete models directly, as risk plus ambiguity (`DiscreteGenerativeModel.expected_free_energy`, see [[docs/api/api_reference|the API reference]]). It does not call RxInfer, and nothing on this page is produced by it.

## Official examples

- Expected free energy minimization: <https://examples.rxinfer.com/categories/advanced_examples/efe_minimization_via_message_passing/>
- T-maze active inference: <https://examples.rxinfer.com/categories/basic_examples/t-maze_active_inference/>
- Active inference mountain car: <https://examples.rxinfer.com/categories/advanced_examples/active_inference_mountain_car/>
