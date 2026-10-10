---
title: Sampling
type: research
status: stable
updated: 2026-10-10
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/importance_sampling]]"
      - "[[docs/api/api_reference]]"
---

# Sampling

`InferenceConfig(method="sampling", ...)` selects sampling-based belief updates;
`num_samples` sets the sample count and `seed` seeds the dispatcher's
`numpy.random.Generator` (default `0`), or pass your own `rng` to
`ActiveInferenceDispatcher`. `SimplePOMDP` and `ContinuousActiveInference` also
take a `seed`. Fix the seed and report it with any stochastic result. Background
on the estimators is in [[knowledge_base/mathematics/importance_sampling]].
