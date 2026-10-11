---
title: Feature Integration
description: Feature integration is the combination of separately coded features such as colour, orientation and motion into unified objects, centred on Treisman's feature integration theory of attention.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - attention
  - binding
  - visual_search
semantic_relations:
  - type: relates
    links:
      - "[[perceptual_binding]]"
      - "[[visual_attention]]"
      - "[[feature_based_attention]]"
      - "[[object_based_attention]]"
aliases:
  - treisman_features
---

# Feature Integration

## Overview
Visual cortex codes features such as colour, orientation, motion and spatial frequency in partly separate maps and pathways. Feature integration is the process by which these are combined so that an observer perceives, for example, a red vertical bar and not a red patch and a vertical edge floating independently. This is one form of the binding problem. The most influential account is Treisman and Gelade's feature integration theory (1980).

## Feature integration theory
The theory proposes two stages:
- **Preattentive stage.** Basic features are registered automatically and in parallel across the visual field in separate feature maps.
- **Attentive stage.** A spatial attention window moves over a master map of locations, and attending to a location binds the features found there into an object representation.

Predictions and evidence:
- **Search slopes.** Finding a target defined by a single feature (a red item among green) shows response time that is nearly flat as the number of distractors grows, consistent with parallel processing. Finding a target defined by a conjunction of features (a red vertical among red horizontal and green vertical items) shows time that rises roughly linearly with display size, consistent with serial attentive search.
- **Illusory conjunctions.** When attention is overloaded or diverted, observers report features recombined incorrectly, such as the colour of one item with the shape of another (Treisman and Schmidt, 1982).
- **Parietal lesions.** Patients with parietal damage show severe binding errors.

Criticisms: many conjunction searches are efficient, and search slopes form a continuum without a clean division into serial and parallel. Guided Search (Wolfe) revises the account so that preattentive feature information guides attention toward likely targets. Alternative accounts emphasise similarity between target and distractors. Other proposals suggest that binding is achieved by synchronised neural firing, a hypothesis that remains debated.

## Role in Active Inference
The framework treats binding as inference: a generative model in which an object causes co-occurring features makes the joint hypothesis the best explanation of the input, and attention is the optimisation of precision, which selects which features and locations drive updating (Feldman and Friston, 2010). Under this view the effects of attention on binding emerge from precision control, and no separate master map is required. This is an interpretive reformulation, and its specific predictions for conjunction search have not been tested against the original theory.

## Related concepts
- [[spatial_attention]] — selection of locations that gates binding
- [[feature_extraction]] — formation of the feature maps
- [[neural_synchrony]] — a proposed binding mechanism
- [[gestalt_principles]] — grouping cues that also determine which features bind
- [[attention_mechanisms]] — the larger set of attentional processes

## References
- Treisman, A. M., & Gelade, G. (1980). A feature-integration theory of attention. Cognitive Psychology.
- Treisman, A., & Schmidt, H. (1982). Illusory conjunctions in the perception of objects. Cognitive Psychology.
- Feldman, H., & Friston, K. J. (2010). Attention, uncertainty, and free-energy. Frontiers in Human Neuroscience.
