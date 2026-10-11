---
title: RxInfer notes
type: documentation
status: stable
updated: 2026-10-10
semantic_relations:
  - type: organizes
    links:
      - "[[AGENTS|RxInfer documentation instructions]]"
      - "[[getting_started|Getting started]]"
      - "[[model_specification|Model specification]]"
      - "[[variational_inference|Variational inference]]"
      - "[[message_passing|Message passing]]"
      - "[[factor_graphs|Factor graphs]]"
      - "[[streaming_inference|Streaming inference]]"
      - "[[systems_engineering_guidebook|Workflow checklist]]"
---

# RxInfer notes

> This folder describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. Official documentation: <https://docs.rxinfer.com/stable/>. Source: <https://github.com/ReactiveBayes/RxInfer.jl>.

RxInfer is a Julia package for Bayesian inference on factor graphs by reactive message passing. The pages here are short summaries of its official documentation, with Julia snippets taken from it. Julia code is not run in this repository. If a page here and the official documentation differ, the official documentation is authoritative.

## Notes

- Basics: [[getting_started]], [[model_specification]], [[model_macro_paradigm]], [[factor_graphs]].
- Inference: [[message_passing]], [[variational_inference]], [[rxinfer_variational_inference]], [[free_energy]], [[execution_engine]].
- Streaming: [[reactive_programming]], [[streaming_inference]].
- Active inference: [[free_energy_message_passing_active_inference]], [[free_energy_message_passing_details]], [[active_inference_examples]], [[rxinfer_mountain_car_readme]].
- Ecosystem and workflow: [[rxinfer_stack]], [[systems_engineering_guidebook]].
- Reduced to pointers (candidates for merging): [[ast_processing]], [[compiler_pipeline]], [[advanced_systems_engineering_guidebook]].

## Official documentation

These pages summarise the RxInfer.jl documentation and do not copy it. The current site is
<https://docs.rxinfer.com/stable/>; it changes with each release, so check it for current
behaviour and API names.

## Python runtime

For the Python package, see the root `README.md` and [[docs/manuscript/README|the manuscript]].
