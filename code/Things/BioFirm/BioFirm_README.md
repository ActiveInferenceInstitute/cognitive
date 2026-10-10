---
title: BioFirm Model Design Notes
type: design_notes
status: draft
created: 2025-01-01
updated: 2026-10-10
tags:
  - biofirm
  - homeostasis
  - pomdp
  - design
semantic_relations:
  - type: relates
    links:
      - "[[code/Things/BioFirm/README|BioFirm folder]]"
      - "[[knowledge_base/BioFirm/biofirm_framework|BioFirm framework]]"
      - "[[knowledge_base/cognitive/homeostatic_control_theory|Homeostatic control theory]]"
      - "[[knowledge_base/mathematics/active_inference_pomdp|Active inference POMDP]]"
---

# BioFirm: Homeostatic Control Model (Design Notes)

> **Status: design sketch; no code in this repository.** `code/Things/BioFirm/` contains documentation only (`README.md`, `AGENTS.md`, this page and `docs/earth_systems_schema.md`). There are no `biofirm_model.py`, `biofirm_environment.py`, `biofirm_analysis.py` or `example.py` files, no simulation runner, no HTML report and no recorded performance numbers. For an executable discrete-state POMDP agent, use `code/Things/Simple_POMDP/`.

## Model specification

BioFirm is specified as a partially observable Markov decision process in which an agent keeps a system inside a homeostatic range.

| Space | Size | Values |
|-------|------|--------|
| Observations | 3 | LOW, MEDIUM, HIGH |
| Hidden states | 5 | TOO_LOW, LOWER_BOUND, MEDIUM, UPPER_BOUND, TOO_HIGH |
| Actions | 3 | DECREASE, MAINTAIN, INCREASE |

The agent receives a coarse observation, infers the finer hidden state, and chooses an action that moves the state toward MEDIUM.

## Generative model

- **A** (observation model): P(o|s), column stochastic, mapping five states to three observations.
- **B** (transitions): P(s'|s,a), one column-stochastic matrix per action.
- **C** (preferences): log-probabilities over observations, strongly favouring MEDIUM.
- **D** (prior): initial belief over hidden states, uniform by default.

## Inference scheme

- **Perception:** minimize variational free energy (accuracy plus complexity) to update beliefs.
- **Action:** compute expected free energy per action (epistemic plus pragmatic value) and select by softmax over negative expected free energy.
- **Learning:** optional adaptation of A, B and C.

See [[knowledge_base/mathematics/variational_free_energy|variational free energy]] and [[knowledge_base/mathematics/expected_free_energy|expected free energy]] for the derivations.

## Quantities worth tracking if implemented

Time in the desired range, mean state deviation, recovery speed after perturbation, belief entropy, belief accuracy against the true state, and action entropy.

## Possible extensions

Hierarchical control over several timescales (see [[code/Things/BioFirm/docs/earth_systems_schema|Earth systems schema]]), continuous state spaces, preference learning, and coupled multi-agent dynamics.

## References

1. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*.
2. Da Costa, L., et al. (2020). Active inference on discrete state-spaces: a synthesis. *Journal of Mathematical Psychology*.
3. Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference. *Biological Cybernetics*.
4. Buckley, C. L., et al. (2017). The free energy principle for action and perception: a mathematical review. *Journal of Mathematical Psychology*.
