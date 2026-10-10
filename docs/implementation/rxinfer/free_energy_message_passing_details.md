---
title: Free energy decomposition
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
      - "[[free_energy_message_passing_active_inference]]"
      - "[[knowledge_base/mathematics/variational_free_energy]]"
---

# Free energy decomposition

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

Standard identities for a variational free energy, independent of any library. For a model `p(x, y) = p(y | x) p(x)` and a distribution `q(x)`:

```text
F[q] = KL[ q(x) || p(x) ] - E_q(x)[ log p(y | x) ]      (complexity minus accuracy)
     = KL[ q(x) || p(x | y) ] - log p(y)                (posterior divergence minus log evidence)
```

so `F` is an upper bound on the negative log evidence and is minimized when `q` equals the posterior. These are textbook results; see [[knowledge_base/mathematics/variational_free_energy]].

RxInfer's documented quantity is the Bethe free energy ([[free_energy]]). It does not export separate "perceptual", "action" or "expected" free energy calculators, and earlier versions of this page that showed such functions and custom message types for active inference described code that is not part of RxInfer. For active inference in RxInfer see [[free_energy_message_passing_active_inference]].

## Official documentation

- Bethe free energy: <https://docs.rxinfer.com/stable/manuals/variational/bethe-free-energy/>
