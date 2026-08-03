---
title: Python FEP Framework
type: implementation_guide
id: fep_python_framework_001
created: 2025-12-18
updated: 2026-08-02
tags:
  - free_energy_principle
  - python
  - implementation
  - variational_inference
  - active_inference
  - machine_learning
aliases: [python_fep, fep_framework, fep_implementation]
semantic_relations:
  - type: implements
    links:
      - [[knowledge_base/free_energy_principle/mathematics/core_principle]]
      - [[knowledge_base/free_energy_principle/mathematics/variational_free_energy]]
      - [[knowledge_base/free_energy_principle/cognitive/perception]]
      - [[knowledge_base/free_energy_principle/biology/homeostasis]]
  - type: foundation
    links:
      - [[knowledge_base/free_energy_principle/implementations/neural_networks]]
      - [[knowledge_base/free_energy_principle/implementations/simulation]]
      - [[knowledge_base/free_energy_principle/AGENTS]]
      - [[knowledge_base/free_energy_principle/mathematics/expected_free_energy]]
  - type: relates
    links:
      - [[knowledge_base/cognitive/active_inference]]
---

# Python FEP Framework

This page records the implementation status of the Free Energy Principle in
Python as it relates to this repository. The repository's own executable
implementation is the `cognitive` package; the FEP mathematics it realizes
is documented under
`knowledge_base/free_energy_principle/mathematics/`.

## The repository's implementation

- **Package**: `cognitive` (source in `code/tools/src`, configured by
  `pyproject.toml`).
- **Discrete models**: `DiscreteGenerativeModel`,
  `ActiveInferenceDispatcher`, `InferenceConfig`, and `ModelState` — the
  canonical example is in the root `README.md`.
- **Executable validation**: `code/tests/` and the manuscript
  (`docs/manuscript/README.md`) document and verify the behavior.
- **Matrix tooling**: `cognitive.models.matrices` (MatrixOps, MatrixLoader,
  MatrixInitializer, MatrixVisualizer).

## Implementation notes

A complete FEP implementation spans likelihood and transition models
(`A`, `B`), preferences (`C`), priors (`D`, `E`), variational belief
updating, expected free energy, and policy selection. This repository
implements the discrete subset with validation; continuous
generalized-coordinate updates are provided by `Things.Continuous_Generic`.

Earlier drafts of this page contained an extensive illustrative `fep`
framework in code blocks. Those blocks described a fictional package that
is not shipped here and not installable; they were removed. Any code shown
in this repository's pages must reference the installed `cognitive` or
`Things` packages (see `knowledge_base/AGENTS.md` for the authoring rules).
