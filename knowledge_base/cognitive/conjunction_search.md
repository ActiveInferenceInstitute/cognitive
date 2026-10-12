---
title: Conjunction Search
description: Visual search for a target defined by a combination of features, why it is typically slower than single-feature search, and how feature integration and guided search theories explain it.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - visual_search
  - attention
  - perception
semantic_relations:
  - type: relates
    links:
      - "[[visual_search]]"
      - "[[feature_integration]]"
      - "[[feature_based_attention]]"
      - "[[priority_maps]]"
      - "[[attention_mechanisms]]"
---

# Conjunction Search

## Overview
In a conjunction search the target differs from the distractors only in a combination of two or more features, for example a red vertical bar among red horizontal and green vertical bars. No single feature identifies it. The task contrasts with feature search, where the target has a unique feature such as being the only red item, and typically pops out.

## Behavioural signature
In classic experiments, response time in feature search is nearly flat as the number of distractors increases, whereas in conjunction search it rises roughly linearly with set size. Treisman and Gelade interpreted the steep slopes as serial, attention-demanding scanning of items, with target-absent trials about twice as steep as target-present trials, consistent with self-terminating search. They also reported illusory conjunctions: when attention is overloaded, features of different objects may be wrongly combined.

## Theoretical accounts
- **Feature integration theory** proposed that features are registered in parallel in separate maps, and that binding them into objects requires focal attention at a location.
- **Guided search** (Wolfe and colleagues) argued that parallel feature processing can steer attention toward items sharing target features, so a conjunction target is found faster than serial scanning of all items would imply. Slopes vary widely across conjunctions, and some, such as motion with form, can be fairly shallow.
- Later work treats search efficiency as a continuum determined by target-distractor similarity and distractor heterogeneity, instead of a strict parallel/serial dichotomy.

## Relation to Active Inference
Active Inference treats visual search as epistemic foraging: eye movements are chosen to resolve uncertainty about the scene, so that salience reflects expected information gain. Simulations of saccadic search have been built within this framework, and top-down feature-based guidance corresponds to precision-weighted priors about the target. This is an active research programme and a conceptual fit, not a demonstration that human search slopes follow from the free energy principle.

## Related concepts
- [[visual_search]] — the general task class
- [[feature_integration]] — the theory that made conjunction search a standard test
- [[feature_based_attention]] — guidance by target features
- [[priority_maps]] — spatial maps combining bottom-up and top-down signals
- [[preattentive]] — the stage at which single features are available in parallel
- [[epistemic_foraging]] — the Active Inference reading of search

## References
- Treisman, A. M., & Gelade, G. (1980). A feature-integration theory of attention. Cognitive Psychology, 12, 97-136.
- Wolfe, J. M., Cave, K. R., & Franzel, S. L. (1989). Guided search: An alternative to the feature integration model for visual search. Journal of Experimental Psychology: Human Perception and Performance, 15, 419-433.
- Wolfe, J. M. (1994). Guided Search 2.0: A revised model of visual search. Psychonomic Bulletin & Review, 1, 202-238.
