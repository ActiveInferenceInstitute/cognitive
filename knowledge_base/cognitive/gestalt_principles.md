---
title: Gestalt Principles
description: Gestalt principles are laws of perceptual organisation, including proximity, similarity, good continuation, closure, common fate and figure-ground, describing how the visual system groups elements into wholes.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - perception
  - grouping
  - vision
semantic_relations:
  - type: relates
    links:
      - "[[perceptual_organization]]"
      - "[[illusory_contours]]"
      - "[[form_perception]]"
      - "[[perceptual_binding]]"
---

# Gestalt Principles

## Overview
Gestalt psychology, developed in the early twentieth century by Wertheimer, Koffka and Kohler, held that perception is organised into structured wholes that are more than the sum of their parts. Its grouping principles describe the conditions under which separate elements are seen as belonging together. Wertheimer set out the main grouping factors in 1923.

## Principles
- **Proximity.** Elements close together are grouped.
- **Similarity.** Elements that share colour, shape or size are grouped.
- **Good continuation.** Elements that lie along a smooth path are grouped, and lines are seen as continuing in their smoothest direction.
- **Closure.** Gaps in a contour are filled in, so incomplete shapes are seen as whole.
- **Common fate.** Elements that move together are grouped.
- **Symmetry and parallelism.** Symmetrical or parallel structure promotes grouping.
- **Figure-ground organisation.** The scene is parsed into a figure that appears in front with a defined contour and a ground that extends behind it.
- **Common region and connectedness.** Elements enclosed in a region or joined by lines are grouped; Palmer and Rock (1994) proposed uniform connectedness as a principle that operates before the classical grouping factors.
- **Pragnanz.** The overarching law that the percept will be as regular, simple and stable as conditions allow.

Gestalt laws were originally demonstrated with informal displays, and the principles are descriptive; they do not specify how strong each is or how they combine when they conflict. Modern work quantifies them in psychophysics, for example measuring the strength of proximity grouping in dot lattices, and investigates neural correlates in the border-ownership and contour-integration responses of visual cortex. Wagemans et al. (2012) review a century of work, including the principles' status and the need for computational models.

## Role in Active Inference
Perceptual organisation is naturally posed as Bayesian inference, with grouping principles reflecting statistical regularities of the natural world: elements that are close, collinear or co-moving are more likely to have a common cause. A generative model that includes objects as causes of correlated features reproduces many of the principles as properties of posterior inference. Illusory contours are then explained as the most probable cause of the visible inducers. In predictive coding implementations, higher areas predict continuations and fill gaps, a proposal consistent with data on contour completion. This is an interpretation: the Gestalt laws were not derived from the free-energy principle, and the simplicity-versus-likelihood debate over Pragnanz is not resolved.

## Related concepts
- [[feature_integration]] — how features bind into objects
- [[visual_perception]] — the system in which grouping occurs
- [[object_recognition]] — recognition builds on grouped shapes
- [[texture_perception]] — segmentation based on similarity
- [[bayesian_brain]] — perception as inference under priors

## References
- Wertheimer, M. (1923). Untersuchungen zur Lehre von der Gestalt II. Psychologische Forschung.
- Palmer, S., & Rock, I. (1994). Rethinking perceptual organization: the role of uniform connectedness. Psychonomic Bulletin and Review.
- Wagemans, J., et al. (2012). A century of Gestalt psychology in visual perception: I. Perceptual grouping and figure-ground organization. Psychological Bulletin.
