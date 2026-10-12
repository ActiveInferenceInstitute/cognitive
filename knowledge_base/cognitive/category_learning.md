---
title: Category Learning
description: How people and models acquire categories from examples and feedback, comparing rule, prototype and exemplar accounts and a Bayesian clustering view, with links to Active Inference.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - learning
  - categorization
  - concepts
semantic_relations:
  - type: relates
    links:
      - "[[pattern_recognition]]"
      - "[[semantic_memory]]"
      - "[[unsupervised_learning]]"
aliases:
  - concept_learning
  - category_formation
wikidata: Q4491893
---

# Category Learning

## Overview
Category learning is the acquisition of the ability to group objects, events or situations into classes and to generalise to new instances. It can be supervised, with feedback on category membership, or unsupervised, where structure is discovered from the statistics of the stimuli (see [[unsupervised_learning]]). It underlies object recognition, concept formation and the organisation of [[semantic_memory]].

## Accounts of category representation
- **Rule-based**: categories are defined by explicit criteria on features. Shepard, Hovland and Jenkins (1961) compared the difficulty of learning different logical category structures, with simple single-dimension rules learned fastest.
- **Prototype**: a category is represented by its central tendency, and membership is graded by similarity to it. Rosch and Mervis (1975) showed that typicality tracks family resemblance, the overlap of features with other category members.
- **Exemplar**: stored instances are compared with a new item, as in the context model of Medin and Schaffer (1978) and the generalised context model of Nosofsky (1986), which adds attention weights over dimensions that are adjusted to maximise discrimination.
- **Clustering and Bayesian**: Anderson (1991) proposed a rational model in which learners infer a latent partition of items into categories, creating a new cluster when data are poorly explained by existing ones.

Evidence supports several representations coexisting, with the dominant strategy depending on the category structure, instruction and individual. Category learning also alters perception: dimensions that are diagnostic become more discriminable (see [[perceptual_learning]]).

## Role in Active Inference
Categories correspond to discrete hidden states of a generative model, and category learning to inference over both assignments and the parameters that map states to features (see [[generative_model]], [[bayesian_learning]]). The rational model of Anderson is a Bayesian mixture-style account with a similar structure, and expanding the number of categories when data are poorly explained is an instance of structure learning (see [[structure_learning]]). Attention weights in exemplar models resemble precision on feature dimensions (see [[precision_weighting]]). In Active Inference proper, the epistemic value of a query provides a rationale for choosing which examples to examine when learning (see [[active_learning]]). These are formal parallels; empirical tests specific to Active Inference models of human category learning are limited.

## Related concepts
- [[learning_theory]] — broader principles of learning
- [[knowledge_organization]] — how learned categories are arranged
- [[feature_extraction]] — features supplied to categorisation

## References
- Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and memorization of classifications. Psychological Monographs, 75(13).
- Rosch, E., & Mervis, C. B. (1975). Family resemblances: Studies in the internal structure of categories. Cognitive Psychology, 7, 573-605.
- Medin, D. L., & Schaffer, M. M. (1978). Context theory of classification learning. Psychological Review, 85, 207-238.
- Nosofsky, R. M. (1986). Attention, similarity, and the identification-categorization relationship. Journal of Experimental Psychology: General, 115, 39-57.
- Anderson, J. R. (1991). The adaptive nature of human categorization. Psychological Review, 98, 409-429.
