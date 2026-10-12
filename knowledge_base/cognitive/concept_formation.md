---
title: Concept Formation
description: How categories and abstract concepts are acquired from experience, classical rule, prototype and exemplar theories, Bayesian accounts, and links to structure learning in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - categorization
  - concepts
  - learning
semantic_relations:
  - type: relates
    links:
      - "[[category_learning]]"
      - "[[prototype_effects]]"
      - "[[schemas]]"
      - "[[generalization]]"
      - "[[structure_learning]]"
---

# Concept Formation

## Overview
Concept formation is the process by which people and other animals form mental representations of categories, such as "bird" or "triangle", from examples, and learn to apply them to new cases. Concepts support recognition, inference, communication and reasoning, since knowing that something belongs to a category licenses predictions about properties that have not been observed (Murphy, 2002). The topic overlaps with [[category_learning]], [[abstraction_levels]] and [[generalization]].

## Theories
- **Rule-based views.** Early studies by Bruner, Goodnow and Austin (1956) asked participants to discover a defining rule by testing hypotheses about which features were relevant, as when a concept is a conjunction of attributes.
- **Prototype theory.** Rosch and Mervis (1975) showed that most natural categories have graded structure with no defining features, and that category members share a family resemblance. Learners are proposed to abstract a central tendency or prototype, and typical items are classified faster (see [[prototype_effects]]).
- **Exemplar theory.** Instead of an abstraction, categories are represented by stored instances, and a new item is classified by similarity to remembered exemplars.
- **Theory-based views.** Background knowledge and causal beliefs constrain which groupings form, so that coherence matters as well as similarity.
- **Bayesian models.** Concept learning has been described as inference over a space of possible concepts, with priors favouring simple hypotheses, which accounts for rapid learning from a few examples (Tenenbaum et al., 2011).

## Development and neural basis
Infants form perceptual categories early, and over development concepts become more abstract and organised in hierarchies, with a basic level that is privileged for naming. Neuroimaging and patient studies implicate the medial temporal lobe, prefrontal cortex and distributed semantic regions, with evidence that different tasks or learning conditions recruit different systems (see [[semantic_memory]] and [[schemas]]). The theories above are not mutually exclusive, and evidence suggests that people can use several strategies.

## Relation to Active Inference
The link is an active research area. Concept learning corresponds to structure learning, in which an agent infers how many latent states or categories to maintain, and compares models that differ in complexity using their evidence (see [[model_comparison]] and [[model_evidence]]). In Active Inference, new latent states can be added when existing ones explain observations poorly, and Bayesian model reduction removes redundant ones. Categories can also be treated as policies and outcome classes that are useful for acting. These formulations are demonstrated mainly in simulations and are not established as accounts of human concept learning.

## Related concepts
- [[category_learning]] — learning to classify items
- [[schemas]] — organised knowledge structures
- [[statistical_learning]] — picking up regularities from experience
- [[categorical_perception]] — perceptual effects of category boundaries
- [[learning_theory]] — broader theories of learning

## References
- Bruner, J. S., Goodnow, J. J., & Austin, G. A. (1956). A study of thinking. Wiley.
- Rosch, E., & Mervis, C. B. (1975). Family resemblances: Studies in the internal structure of categories. Cognitive Psychology, 7, 573-605.
- Murphy, G. L. (2002). The big book of concepts. MIT Press.
- Tenenbaum, J. B., Kemp, C., Griffiths, T. L., & Goodman, N. D. (2011). How to grow a mind: Statistics, structure, and abstraction. Science, 331, 1279-1285.
