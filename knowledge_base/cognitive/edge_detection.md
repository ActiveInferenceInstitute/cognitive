---
title: Edge Detection
description: The early visual process of locating boundaries where image intensity changes sharply, covering receptive-field models in visual cortex, computational formulations, and links to hierarchical predictive processing.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - vision
  - early_vision
  - neural_coding
semantic_relations:
  - type: relates
    links:
      - "[[visual_perception]]"
      - "[[neural_coding]]"
      - "[[perceptual_organization]]"
      - "[[form_perception]]"
      - "[[predictive_coding]]"
---

# Edge Detection

## Overview
Edge detection is the extraction of locations where image properties such as luminance, colour or texture change abruptly. Such discontinuities often coincide with the boundaries of objects, shadows and surface markings, so they carry much of the information needed to describe a scene. Edge detection is studied both as a stage of biological vision and as a problem in computational vision (see [[computational_vision]]).

## Biological basis
Retinal ganglion cells and neurons in the lateral geniculate nucleus have centre-surround receptive fields that respond to local contrast and are largely insensitive to uniform illumination. In primary visual cortex, Hubel and Wiesel (1962) found simple cells that respond best to bars or edges of a particular orientation at a particular position, and complex cells that tolerate small shifts in position. Subsequent work described these cells with oriented, band-pass filters, often modelled as Gabor functions. Lateral interactions between neighbouring cells enhance contrast at boundaries (see [[lateral_connections]]), and later stages integrate local edge responses into contours (see [[perceptual_organization]]).

## Computational formulations
Marr and Hildreth (1980) proposed that edges correspond to zero-crossings of a smoothed second derivative of the image, computed at several spatial scales. Canny (1986) formulated edge detection as an optimisation problem with criteria of good detection, accurate localisation and a single response per edge, leading to a filter based on the derivative of a Gaussian followed by non-maximum suppression and thresholding. Both approaches share the idea of smoothing to suppress noise and then locating peaks of the intensity gradient.

## Relation to Active Inference
Edge-selective responses are a standard example in accounts of predictive coding. Sparse coding of natural images yields oriented, localised filters resembling simple-cell receptive fields, and hierarchical models propose that higher areas predict the activity of lower ones, with edges signalled where prediction fails (see [[predictive_coding]], [[prediction_error]]). This is an active research area with supporting models and partial neural evidence, not a settled mechanism. Within Active Inference, edges are low-level features of the generative model's mapping from latent causes to sensory data, and eye movements that sample informative boundaries are consistent with action chosen to reduce uncertainty.

## Related concepts
- [[visual_perception]] — the broader processing context for early vision
- [[neural_coding]] — how neuron populations represent image features
- [[contrast_sensitivity]] — sensitivity that underlies detection of boundaries
- [[form_perception]] — grouping edges into shapes
- [[predictive_coding]] — hierarchical prediction of low-level features

## References
- Hubel, D. H., & Wiesel, T. N. (1962). Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. Journal of Physiology, 160, 106-154.
- Marr, D., & Hildreth, E. (1980). Theory of edge detection. Proceedings of the Royal Society of London B, 207, 187-217.
- Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 8, 679-698.
