---
title: Feature Extraction
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - perception
  - representation
  - computation
aliases:
  - feature_detection
semantic_relations:
  - type: relates
    links:
      - "[[pattern_recognition]]"
      - "[[sensory_processing]]"
      - "[[hierarchical_processing]]"
wikidata: Q1026626
---

# Feature Extraction

## Overview
Feature extraction is the transformation of raw sensory input into a smaller or more informative set of descriptors, called features, that make later tasks such as recognition, grouping or control easier. In biological systems it is carried out by successive neural stages; in machine perception it is either designed by hand or learned from data.

## Mechanism
Early visual processing is the standard example. Retinal and LGN cells compute local contrast with center-surround filters. Simple cells in V1 behave approximately like oriented, localized, band-pass filters, often modeled by Gabor functions; complex cells pool over such filters to gain tolerance to small shifts. Later areas combine them into responses to corners, curvature, textures and parts. The same pattern, filtering followed by nonlinearity and pooling, defines convolutional networks.

Handcrafted pipelines (edge detectors, SIFT, histograms of oriented gradients) fixed the features in advance. Learned feature extractors instead derive them from the statistics of the data. Principal component analysis finds decorrelating directions; independent component analysis and sparse coding find localized oriented filters on natural images resembling V1 receptive fields (see [[sparse_coding]]); deep networks learn feature hierarchies end to end.

Good features trade off several demands: they should be informative about the quantities that matter, invariant to nuisance variation, and compact. Which features are useful depends on the task, so extraction is not neutral; task-dependent selection can occur through attention ([[feature_based_attention]]).

## Role in Active Inference
In a generative model, the counterpart of feature extraction is inference of latent causes: hidden states at one level are the "features" that explain activity at the level below, and the likelihood mapping specifies how states generate observations. Hierarchical predictive coding extracts features implicitly by minimizing prediction error, with higher levels representing more abstract and slower causes. Learning of the likelihood mapping then plays the role that filter learning plays in unsupervised feature learning.

The distinction matters: feature extraction in the feedforward sense is a bottom-up mapping, while the inferential view is a bidirectional process in which what counts as a feature is set partly by expectations. Whether biological features are better described by one or the other is an open empirical question.

## Related concepts
- [[pattern_recognition]] — downstream use of features
- [[object_recognition]] — recognition built on extracted features
- [[sensory_processing]] — general framing of early processing
- [[sparse_coding]] — learned feature dictionaries for natural signals
- [[computational_vision]] — models that implement feature extraction
- [[hierarchical_models]] — layered abstraction of features

## References
- Hubel, D. H., & Wiesel, T. N. (1962). Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. Journal of Physiology.
- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature.
