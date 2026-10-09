---
title: Visual Attention
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - attention
  - vision
  - precision
semantic_relations:
  - type: relates
    links:
      - "[[attention_mechanisms]]"
      - "[[precision_weighting]]"
      - "[[active_inference]]"
---

# Visual Attention

## Overview
Visual attention is the set of processes that select some visual information for enhanced processing at the expense of the rest. Selection can be by location (spatial attention), by feature such as color or motion, or by whole object. It can be driven by task goals (endogenous) or by salient stimuli (exogenous), and it can be overt, with eye movements, or covert, without them.

## Mechanism
Posner's cueing paradigm showed that a valid spatial cue speeds responses and improves detection at the cued location, without eye movement. Treisman and Gelade's feature-integration theory explained serial versus parallel search: single features are registered in parallel, while conjunctions need attention to bind them. Desimone and Duncan's biased-competition account holds that objects compete for neural representation and attention biases the competition toward task-relevant items.

In visual cortex, attending to a stimulus typically increases firing of neurons responding to it, sharpens or shifts their tuning, and reduces noise correlations. Normalization models, such as the one proposed by Reynolds and Heeger, express these effects as a change in the attentional gain applied before divisive normalization (see [[gain_control]]). Salience models, such as Itti and Koch's, compute a bottom-up priority map from local feature contrasts; top-down control from frontal and parietal regions combines with it.

## Role in Active Inference
Feldman and Friston (2010) proposed that attention is the optimization of precision: increasing the expected precision of prediction errors from an attended channel gives them more influence on belief updating. This maps spatial and feature-based attention onto [[precision_weighting]], and explains why attention can be framed as a form of inference about the reliability of sensory data. Overt attention fits naturally: under [[active_inference]], saccades are selected to sample locations with high expected information gain, so looking is itself an action chosen to reduce uncertainty.

The caveat is that precision-as-attention is a theory that unifies several findings, not a derived consequence of the physiology. Biased competition and normalization models are mechanistically specific and make overlapping predictions, so the data often do not discriminate between them.

## Related concepts
- [[attention_mechanisms]] — general computational and neural accounts of attention
- [[feature_based_attention]] — selection by feature rather than location
- [[object_based_attention]] — selection of whole objects
- [[endogenous_attention]] — goal-driven control
- [[exogenous_attention]] — stimulus-driven capture
- [[perception_attention]] — relation between perception and attention

## References
- Posner, M. I. (1980). Orienting of attention. Quarterly Journal of Experimental Psychology.
- Treisman, A. M., & Gelade, G. (1980). A feature-integration theory of attention. Cognitive Psychology.
- Desimone, R., & Duncan, J. (1995). Neural mechanisms of selective visual attention. Annual Review of Neuroscience.
- Feldman, H., & Friston, K. J. (2010). Attention, uncertainty, and free-energy. Frontiers in Human Neuroscience.
