---
title: Generic POMDP Design Notes
type: design_notes
status: draft
created: 2025-01-01
updated: 2026-10-10
tags:
  - pomdp
  - active_inference
  - expected_free_energy
  - design
semantic_relations:
  - type: relates
    links:
      - "[[code/Things/Generic_POMDP/README|Generic POMDP folder]]"
      - "[[knowledge_base/agents/GenericPOMDP/agent_config|Agent configuration]]"
      - "[[knowledge_base/agents/GenericPOMDP/matrices/A_matrix|A matrix]]"
      - "[[knowledge_base/mathematics/active_inference_pomdp|Active inference POMDP]]"
      - "[[knowledge_base/mathematics/expected_free_energy|Expected free energy]]"
---

# Generic POMDP: Design Notes

> **Status: design sketch; no code in this repository.** `code/Things/Generic_POMDP/` contains documentation only (`README.md`, `AGENTS.md`, this page). There is no `generic_pomdp` module, no `GenericPOMDP` class, no `test_generic_pomdp.py`, no state save/load and no visualization code here. The executable discrete Active Inference agent is `code/Things/Simple_POMDP/` (see [[docs/api/api_reference|API reference]]). The knowledge-base pages under `knowledge_base/agents/GenericPOMDP/` document the intended matrices and configuration.

## Matrix conventions

| Matrix | Shape | Meaning |
|--------|-------|---------|
| A (observation model) | `(num_observations, num_states)` | P(o\|s), columns sum to 1 |
| B (transitions) | `(num_states, num_states, num_actions)` | P(s'\|s,a), column stochastic per action |
| C (preferences) | `(num_observations, planning_horizon)` | log-probabilities of preferred observations |
| D (initial beliefs) | `(num_states,)` | prior over states |
| E (policy prior) | `(num_actions,)` | prior over actions |

## Expected free energy terms

A policy is scored with three terms:

1. **Ambiguity:** expected uncertainty of observations given states; low ambiguity favours information-seeking.
2. **Risk:** KL divergence between predicted and preferred observations.
3. **Expected preference:** value of anticipated observations under C.

Action selection enumerates policies up to the planning horizon, computes these terms, applies a softmax with a temperature, and samples from the result. See [[knowledge_base/mathematics/expected_free_energy|expected free energy]].

## Numerical practice worth following in any implementation

Floor probabilities at a small positive value, use a safe logarithm, normalize intermediate quantities, and bound preference values.

## Related

- [[knowledge_base/agents/GenericPOMDP/README|Generic POMDP knowledge base]]
- `code/Things/Simple_POMDP/` for runnable code and tests.

## References

- Friston, K., et al. (2017). Active inference: a process theory. *Neural Computation*.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*.
- Tishby, N., & Polani, D. (2011). Information theory of decisions and actions.
