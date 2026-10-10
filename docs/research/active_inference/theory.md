---
title: Active inference theory in this repository
type: research
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/mathematics/active_inference_pomdp]]"
      - "[[knowledge_base/mathematics/expected_free_energy]]"
      - "[[knowledge_base/cognitive/free_energy_principle]]"
---

# Active inference theory in this repository

The package implements the discrete-state, discrete-time formulation, plus a
continuous-state agent in generalized coordinates.

## Discrete formulation

A generative model is the tuple `A, B, C, D, E` with `A[o, s] = P(o | s)`,
`B[s_next, s_prev, a]`, log-preferences `C`, state prior `D`, and action prior
`E` (`DiscreteGenerativeModel`). Beliefs are updated by Bayes' rule over the
predicted state distribution. A policy is scored by expected free energy, which
the package defines as risk plus ambiguity. The expected information gain is
reported as a diagnostic and is already contained in the ambiguity term; adding
it again would double count (see `CHANGELOG.md`, release 1.1.0).

The dispatcher turns policy energies into a distribution over first actions
either by taking the best continuation of each action (`PolicyAggregation.MIN`)
or by a Bayesian model average (`PolicyAggregation.MARGINAL`).

## Continuous formulation

`Things.Continuous_Generic.ContinuousActiveInference` holds beliefs over
generalized coordinates and returns an action and a free energy at each step.

## Where the theory is written down

[[knowledge_base/cognitive/active_inference]],
[[knowledge_base/mathematics/active_inference_pomdp]],
[[knowledge_base/mathematics/expected_free_energy]],
[[knowledge_base/mathematics/variational_free_energy]], and
[[knowledge_base/cognitive/free_energy_principle]]. The signatures are in
[[docs/api/api_reference|the API reference]].
