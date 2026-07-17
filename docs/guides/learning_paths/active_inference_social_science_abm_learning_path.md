---
title: Active inference for social-science modelling
type: learning_path
status: stable
---

# Active inference for social-science modelling

This learning path separates theory from the executable repository surface.
For categorical experiments, instantiate `DiscreteGenerativeModel` with
explicit `A`, `B`, `C`, `D`, and `E` arrays and pass it to
`ActiveInferenceDispatcher`. For continuous experiments, use
`ContinuousActiveInference` with validated dynamics and observation matrices.

## Study sequence

1. Read `knowledge_base/cognitive/active_inference_agent.md` for the
   generative-model vocabulary.
2. Read `docs/manuscript/01_introduction.md` and
   `docs/manuscript/02_methodology.md` for the repository's formal contract.
3. Run the discrete example in the root `README.md`.
4. Extend the configuration and add a real regression test before changing a
   model dimension or policy horizon.
5. Use `cognitive-benchmark` for timing diagnostics and
   `cognitive-build-manuscript` for reproducible figures and reporting.

## Interpretation

Social-science applications may add observations, priors, and domain-specific
metrics, but the implementation must preserve normalized beliefs and explicit
configuration. A domain model should report empirical data provenance,
parameter choices, uncertainty, and a comparison protocol. The repository's
runtime does not provide a social-agent framework; it provides validated
inference components that can be composed by a domain project.
