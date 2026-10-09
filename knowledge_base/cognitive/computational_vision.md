---
title: Computational Vision
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - vision
  - computation
  - modeling
semantic_relations:
  - type: relates
    links:
      - "[[visual_perception]]"
      - "[[generative_model]]"
      - "[[bayesian_inference]]"
---

# Computational Vision

## Overview
Computational vision is the study of vision as an information-processing problem: what must be computed from images to recover properties of the world, and by what algorithms and circuits it can be done. It serves both biological vision, as a source of models, and machine vision, as an engineering discipline.

## Mechanism
Marr's framework separates three levels of analysis: the computational theory (what is computed and why), the algorithm and representation, and the physical implementation. His proposed stages move from an image to a primal sketch of edges and blobs, to a viewer-centered 2.5D sketch of surface orientation and depth, to an object-centered 3D model.

Vision is typically cast as an inverse problem. Image formation (optics, shading, projection) maps scene properties to pixels, and vision must invert it from data that underdetermine the answer. Regularization and prior knowledge select among solutions. The Bayesian formulation makes this explicit: infer scene variables $S$ from image $I$ using $p(S\mid I)\propto p(I\mid S)\,p(S)$. A related program, analysis by synthesis, evaluates hypotheses by generating the image they would produce and comparing it with the observed one (Yuille and Kersten).

Alternative approaches skip explicit inversion. Discriminative models, now mainly deep convolutional networks, learn a direct mapping from image to label or scene property from large datasets, and perform strongly in practice. They typically lack an explicit generative account and can fail under distribution shift in ways human vision does not.

Standard subproblems include edge and feature detection, stereo and depth, motion estimation (optical flow), segmentation, shape from shading, object recognition and tracking.

## Role in Active Inference
Generative accounts of vision are the closest computational relative of [[active_inference]]: a hierarchical generative model of images, inverted by approximate Bayesian inference, with top-down predictions compared against input. Rao and Ballard's predictive coding model is the canonical biological instance. Active inference adds that the sensor is controllable, so eye and camera movements are chosen to resolve ambiguity, an element missing from passive image-to-label systems. Practical large-scale vision systems are still dominated by discriminative learning, so the generative framework's engineering advantage is not established.

## Related concepts
- [[visual_perception]] — the perceptual capacity being modeled
- [[object_recognition]] — a central task
- [[feature_extraction]] — early representational stage
- [[depth_perception]] — a classic inverse problem
- [[motion_perception]] — motion estimation and its perception
- [[predictive_coding]] — a generative-model-based vision scheme

## References
- Marr, D. (1982). Vision. W. H. Freeman.
- Yuille, A., & Kersten, D. (2006). Vision as Bayesian inference: analysis by synthesis? Trends in Cognitive Sciences.
- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex. Nature Neuroscience.
