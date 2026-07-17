---
title: Active inference learning path
type: learning_path
status: stable
---

# Active inference learning path

## Foundations

Study probability, Bayesian filtering, information theory, and the
free-energy principle. The citations and formal definitions used by this
repository are collected in `docs/manuscript/07_scope_and_related_work.md`.

## Executable sequence

1. Install the package with `python -m pip install -e ".[dev]"`.
2. Construct a `DiscreteGenerativeModel` with validated matrices.
3. Run `ActiveInferenceDispatcher` with `variational`, `mean_field`, and
   `sampling` methods.
4. Compare the policy distributions and expected-free-energy components.
5. Study continuous generalized coordinates with
   `ContinuousActiveInference`.
6. Build the manuscript and inspect the generated figures and PDF.

## Extensions

An extension should define its state and observation semantics, provide shape
and probability checks, accept explicit random seeds, persist state through a
versioned schema, and add negative controls. The package does not expose a
general environment, social-agent, or neural-network framework; those belong
in a domain project that can depend on this package.
