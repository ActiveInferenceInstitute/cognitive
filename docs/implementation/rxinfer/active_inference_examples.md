---
title: Active inference examples with RxInfer
type: documentation
status: stable
created: 2024-03-20
updated: 2026-10-10
tags:
  - rxinfer
  - julia
  - active-inference
semantic_relations:
  - type: relates
    links:
      - "[[free_energy_message_passing_active_inference]]"
      - "[[rxinfer_mountain_car_readme]]"
      - "[[getting_started]]"
---

# Active inference examples with RxInfer

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

The RxInfer examples site hosts maintained active inference examples. They are the reference for how agents are written with `@model`, custom nodes and `infer`; this repository does not reproduce them.

| Example | Category |
|---|---|
| T-maze active inference | basic |
| POMDP control | basic |
| Active inference mountain car | advanced |
| Autoregressive active inference | advanced |
| Expected free energy minimization via message passing | advanced |
| Multi-agent trajectory planning | advanced |

See <https://examples.rxinfer.com/> for the list and the code. Earlier versions of this page showed generic `@model` sketches for decision making and hierarchical perception-action that are not taken from those examples and were removed.

For the discrete expected free energy used by this repository's Python package, see [[docs/api/api_reference|the API reference]].

## Official examples

- T-maze: <https://examples.rxinfer.com/categories/basic_examples/t-maze_active_inference/>
- POMDP control: <https://examples.rxinfer.com/categories/basic_examples/pomdp_control/>
- Mountain car: <https://examples.rxinfer.com/categories/advanced_examples/active_inference_mountain_car/>
- Expected free energy minimization: <https://examples.rxinfer.com/categories/advanced_examples/efe_minimization_via_message_passing/>
