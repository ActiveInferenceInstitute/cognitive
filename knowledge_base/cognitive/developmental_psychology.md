---
title: Developmental Psychology
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - cognitive
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/cognitive/learning_theory]]"
      - "[[knowledge_base/cognitive/cognitive_development]]"
      - "[[knowledge_base/cognitive/epistemic_foraging]]"
---

# Developmental Psychology

## Overview

Developmental psychology studies how perception, action, language, social understanding and reasoning change from infancy onward. A computational reading, close to the "child as scientist" idea in the Bayesian learning literature, treats development as the construction and revision of a generative model of the world. Active Inference supplies the objective for that revision. The idea is a plausible and active research programme; fitted Active Inference models of specific developmental data sets are still few.

## Main Ideas

**Learning as model building.** Over development the child must acquire the structure of the model (which hidden states exist and how they relate) as well as its parameters. Parameter learning corresponds to updating the likelihood and transition matrices from experience. Structure learning is harder; Bayesian model reduction is one proposed mechanism, in which an over-flexible model is pruned of components that do not improve evidence (Smith, Friston and Whyte 2022 give a tutorial treatment). It is offered as a candidate mechanism for how a model comes to fit the world with fewer parameters, not as a demonstrated account of any developmental transition.

**Curiosity as epistemic value.** An agent minimising expected free energy is attracted to actions that reduce uncertainty about the world. This gives a principled reading of infant exploration, play and the attention children direct at events that are neither predictable nor chaotic ([[knowledge_base/cognitive/epistemic_foraging]]). The empirical anchor is the looking-time literature: infants look longer at events that violate their expectations, which is consistent with surprise-driven attention, although looking time alone does not identify the underlying computation.

**Stages as changes in model depth.** Piagetian or later stage accounts can be redescribed as growth in hierarchical depth and temporal horizon of the generative model ([[knowledge_base/cognitive/hierarchical_models]]). This is an analogy: it reframes known phenomena without yet predicting new ones.

**Plasticity and priors.** Early in life priors are weak and learning rates high; later priors harden. This echoes critical periods and the stability-plasticity trade-off ([[knowledge_base/cognitive/stability_plasticity]], [[knowledge_base/cognitive/neural_plasticity]]).

## Caveats

Development is also biological maturation, and it unfolds under social scaffolding. A free-energy account says little by itself about maturational timing or about why particular abilities emerge in a particular order, so it complements instead of replacing the empirical literature.

## In This Repository

Related pages are [[knowledge_base/cognitive/cognitive_development]], [[knowledge_base/cognitive/cognitive_development_ai]], [[knowledge_base/cognitive/meta_learning]], [[knowledge_base/cognitive/statistical_learning]] and the biological counterpart [[knowledge_base/biology/developmental_systems]]. The agent code fixes the generative model's structure at construction time (`DiscreteGenerativeModel` in `code/tools/src/models/active_inference`), so the codebase contains no structure learning of the kind described above.

## References

- Smith, R., Friston, K. J., Whyte, C. J. (2022). A step-by-step tutorial on active inference and its application to empirical data. *Journal of Mathematical Psychology*.

## Related Resources

- [[knowledge_base/cognitive/active_inference]]
- [[knowledge_base/cognitive/learning_theory]]
