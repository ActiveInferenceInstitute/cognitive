---
title: Research documentation
type: guide
status: stable
created: 2024-02-06
updated: 2026-10-10
tags:
  - research
  - theory
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/cognitive/cognitive_modeling_concepts]]"
      - "[[knowledge_base/mathematics/active_inference_pomdp]]"
      - "[[docs/api/api_reference]]"
---

# Research documentation

This page maps each research topic to the place where it is actually documented
or implemented. The package computes the discrete-state quantities below; the
rest is theory in the knowledge base.

## Where each topic lives

| Topic | Theory | Executable in this repository |
|---|---|---|
| POMDP generative model (A, B, C, D, E) | [[knowledge_base/mathematics/active_inference_pomdp]], [[knowledge_base/mathematics/pomdp_structure]] | `DiscreteGenerativeModel` |
| Belief updating | [[knowledge_base/mathematics/belief_updating]] | `DiscreteGenerativeModel.posterior`, `ActiveInferenceDispatcher.dispatch_belief_update` |
| Expected free energy | [[knowledge_base/mathematics/expected_free_energy]], [[knowledge_base/mathematics/efe_components]] | `DiscreteGenerativeModel.expected_free_energy` (total = risk + ambiguity) |
| Policy selection | [[knowledge_base/mathematics/policy_selection]] | `ActiveInferenceDispatcher.dispatch_policy_inference`, `enumerate_policies`, `evaluate_policy` |
| Variational and sampling inference | [[knowledge_base/mathematics/variational_inference]] | `InferenceMethod` values `variational`, `sampling`, `mean_field` |
| Continuous generalized coordinates | [[knowledge_base/mathematics/continuous_time_models]] | `Things.Continuous_Generic.ContinuousActiveInference` |
| Homeostatic control | [[knowledge_base/cognitive/homeostatic_control_theory]] | `HomeostaticInference`, `HomeostaticFactory` |
| Hierarchical models | [[knowledge_base/cognitive/hierarchical_models]] | not built; `policy_type="hierarchical"` raises `ValueError` |
| Multi-agent and swarm systems | [[knowledge_base/cognitive/multi_agent_active_inference]], [[knowledge_base/cognitive/swarm_intelligence]] | not built; `code/Things/Ant_Colony/` and `code/Things/KG_Multi_Agent/` contain documentation only |

## How results are produced

- Unit and property checks: `python -m pytest -q` (the suite is in `code/tests/`;
  `test_math_oracles.py` compares the package against independent calculations).
- Runtime numbers: `cognitive-benchmark` writes machine-readable timings.
- Manuscript numbers: `cognitive-build-manuscript` injects values from the build
  into the text; see [[docs/manuscript/06_reproducibility|the reproducibility chapter]].

There is no research-specific API; see [[docs/api/research_api|Research API]].

## Related documentation

- [[docs/implementation/implementation_guides|Implementation guides]]
- [[docs/api/api_documentation|API documentation]]
- [[docs/examples/usage_examples|Usage examples]]
- [[docs/guides/research_guides|Research guides]]
