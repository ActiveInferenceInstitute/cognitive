---
title: Free energy in RxInfer
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
      - "[[variational_inference]]"
      - "[[message_passing]]"
      - "[[free_energy_message_passing_details]]"
---

# Free energy in RxInfer

> This page describes an external project, RxInfer.jl, which is not part of this repository and is not used by the Python package. It summarizes the official documentation at <https://docs.rxinfer.com/stable/>; if the two differ, the official documentation is authoritative. Julia snippets on this page are not run here.

The variational free energy of a distribution `q` over latent variables `x` for observed data `y` is

```text
F[q](y) = E_q(x)[ log q(x) / f(y, x) ]
```

where `f` is the factorized model. Minimizing it over a family of distributions gives an approximation of the posterior.

## Bethe approximation

RxInfer restricts `q` to the Bethe form, built from the factorization of the model itself:

```text
q(x) = product over a of q_a(x_a)  /  product over i of q_i(x_i)^(d_i - 1)
```

where `d_i` is the number of factors that variable `i` appears in. Substituting into the free energy gives the Bethe free energy,

```text
F_B = sum_a U_a[q_a] - sum_a H[q_a] + sum_i (d_i - 1) H[q_i]
```

with average energy `U_a = -E_qa[log f_a]` and entropies `H`.

## What RxInfer computes

The engine optimizes the Bethe free energy implicitly through message passing and does not evaluate its value unless asked: `infer` has a `free_energy` flag. Because messages arrive reactively, a computed value can lag behind the true state, especially in models with loops, so the documentation advises checking that the value converges rather than that it decreases monotonically. When the value is requested, RxInfer raises an error if a local computation produces `NaN` or `Inf`.

This is not the expected free energy of active inference; see [[free_energy_message_passing_active_inference]].

## Official documentation

- Bethe free energy: <https://docs.rxinfer.com/stable/manuals/variational/bethe-free-energy/>
- Further reading cited there: Yedidia et al. (2005) on belief propagation and region-based approximations; Dauwels (2007) on variational message passing on Forney-style factor graphs; Senoz et al. (2021), <https://doi.org/10.3390/e23070807>.
