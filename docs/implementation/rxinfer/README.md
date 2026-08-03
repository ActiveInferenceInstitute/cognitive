---
title: RxInfer integration index
type: documentation
status: stable
semantic_relations:
  - type: organizes
    links:
      - '[[AGENTS|RxInfer documentation instructions]]'
      - '[[getting_started|Getting started]]'
      - '[[model_specification|Model specification]]'
      - '[[variational_inference|Variational inference]]'
      - '[[message_passing|Message passing]]'
      - '[[factor_graphs|Factor graphs]]'
      - '[[streaming_inference|Streaming inference]]'
      - '[[systems_engineering_guidebook|Systems engineering guidebook]]'
---

# RxInfer integration index

This subtree documents the RxInfer (Julia) probabilistic-programming
integration for message-passing and variational-inference models.

## Local notes

- `getting_started.md` — first steps with the integration.
- `model_specification.md` and `model_macro_paradigm.md` — model definition.
- `variational_inference.md`, `rxinfer_variational_inference.md`,
  `free_energy.md`, `free_energy_message_passing_active_inference.md`,
  `free_energy_message_passing_details.md` — inference mathematics.
- `message_passing.md`, `factor_graphs.md`, `reactive_programming.md`,
  `streaming_inference.md`, `execution_engine.md`, `ast_processing.md`,
  `compiler_pipeline.md` — engine concepts.
- `rxinfer_stack.md` — the integration stack.
- `systems_engineering_guidebook.md` and
  `advanced_systems_engineering_guidebook.md` — engineering guides.
- `active_inference_examples.md` and `rxinfer_mountain_car_readme.md` —
  worked examples.

## Vendored documentation

`docs/` is a read-only snapshot of the RxInfer documentation (Documenter.jl
site). See `AGENTS.md` for the maintenance rule.

## Notes

- The integration is Julia software; these pages are not part of the Python
  package and are excluded from its linters.
- For the Python runtime, see the root `README.md` and
  `docs/manuscript/README.md`.
