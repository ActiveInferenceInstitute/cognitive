---
title: Scientific Computing
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - reproducibility
  - software
  - overview
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/computational_methods]]"
      - "[[knowledge_base/mathematics/numerical_methods]]"
      - "[[knowledge_base/mathematics/statistical_computing_implementations]]"
      - "[[knowledge_base/research/concepts/validation_methods]]"
---

# Scientific Computing

## Overview
Scientific computing is the practice of implementing mathematical models as software so that results are correct, reproducible and inspectable. For this repository that means treating the knowledge base's formulas as specifications and the Python package as an executable check on them.

## Definition / Formalism
Four properties matter, and each has a concrete counterpart in the repo.

**Correctness against an independent reference.** A result that is well-formed can still be wrong. `code/tests/test_math_oracles.py` compares implementations to quantities derived separately: analytic values, algebraically equivalent decompositions, finite-difference gradients. One test pins the identity that the two expected-free-energy decompositions agree, $\text{risk}+\text{ambiguity}=\text{expected surprise}-\text{information gain}$, over randomly generated models of several shapes. That identity is the one at stake in the "Corrected mathematics" entry of `CHANGELOG.md`, where a double-counted epistemic term was removed from the objective.

**Input validation.** `DiscreteGenerativeModel` rejects non-finite or negative entries, non-column-stochastic `A` and `B`, and mismatched dimensions at construction. `MatrixOps` helpers validate finite, non-empty numeric input rather than silently coercing it.

**Determinism.** Randomness is routed through explicit `numpy.random.Generator` objects: `MatrixInitializer.random_stochastic(shape, rng=...)` takes one, the dispatcher's `InferenceConfig` carries a `seed` (default 0), and the benchmarks fix seed 0.

**Measured quality gates.** `pyproject.toml` turns on branch coverage over the whole source set with a `fail_under` floor of 90, and configures `ruff` and `mypy` as dev tools. The comments there record that an earlier include filter overstated coverage (93% reported against 70% measured), which is why filters are not to be added back.

## Role in Active Inference
Active inference models are small linear-algebra programs wrapped around probabilistic identities, so the main risks are silent normalisation errors, orientation mistakes (rows versus columns) and conflated formulas. The repository's convention is that probabilities are column-stochastic: `A[o, s] = P(o | s)`, `B[s_next, s_prev, a]`. Mixing orientations would still produce plausible-looking numbers, which is why identity-based tests are preferable to range checks alone.

The `cognitive-benchmark` console command (`--repetitions N`, optional `--output path.json`) emits a JSON report of per-call timings for softmax, dispatcher belief update, a Simple POMDP step and a continuous-inference step.

## Related concepts
- [[computational_methods]] — algorithms and floating-point care
- [[numerical_methods]] — stable softmax and log-sum-exp
- [[matrix_operations]] — the conventions the tests enforce
- [[statistical_computing_implementations]] — worked statistical code
- [[validation_methods]] — model-level validation alongside code-level tests
- [[expected_free_energy]] — the quantity the oracle tests protect
