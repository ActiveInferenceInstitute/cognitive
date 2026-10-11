---
title: Size Constancy
description: The perceptual stability of an object's apparent size across changes in viewing distance, covering size-distance scaling, its limits, classic illusions, and an inferential interpretation.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - perception
  - vision
  - depth
semantic_relations:
  - type: relates
    links:
      - "[[perceptual_constancy]]"
      - "[[shape_constancy]]"
      - "[[depth_perception]]"
      - "[[perceptual_inference]]"
---

# Size Constancy

## Overview
Size constancy is the tendency to perceive an object as having a fixed size when its distance from the observer changes. A person walking away occupies a smaller and smaller region of the retina, yet is not seen as shrinking. The retinal size of an object is proportional to its physical size divided by its distance, so a given image size is compatible with many combinations of size and distance. Perceived size depends on how that ambiguity is resolved. It is one form of [[perceptual_constancy]].

## Size-distance scaling
A classical account is size-distance invariance: perceived size is proportional to retinal size multiplied by perceived distance. This relation is often associated with Emmert's law, which describes how the apparent size of an afterimage grows with the distance of the surface on which it is projected. The account makes a testable prediction, that misjudging distance should produce a matching error in size. Several effects are commonly explained this way:
- The **Ponzo illusion**, in which converging lines make an upper bar appear larger because it seems farther away.
- The **Ames room**, in which a distorted room produces an image consistent with a normal one, and the people in it look unequal in size.
- The **moon illusion**, though its explanation remains debated and apparent-distance accounts are only one proposal.

Constancy is strong when depth information is rich, as in the real environment, and declines when cues are removed, for example in a dark room with isolated luminous targets (see [[depth_perception]]). Studies of apparent size at varied distances show a compromise between the retinal and the physical size in many conditions.

## Relation to Active Inference
Size constancy has a natural reading as inference about a latent size and distance from a single ambiguous image quantity. A generative model that couples object size, distance and image size supports a posterior over size once distance is constrained by cues such as familiar size, texture gradients, disparity and occlusion (see [[perceptual_inference]], [[generative_model]]). Weighting the cues by reliability corresponds to precision weighting (see [[precision_weighting]]). Illusions follow when the model assumes depth structure that the stimulus does not have. This is an application of general Bayesian perceptual principles, not a distinct result of Active Inference, and the exact computations used by the visual system are still being studied.

## Related concepts
- [[perceptual_constancy]] — the general phenomenon of stable percepts
- [[shape_constancy]] — the parallel case for perceived shape
- [[depth_perception]] — cues that fix distance
- [[visual_perception]] — the wider processing stream
- [[perceptual_inference]] — the inferential framework

## References
- Walsh, V., & Kulikowski, J. (Eds.). (1998). Perceptual constancy: Why things look as they do. Cambridge University Press.
- Palmer, S. E. (1999). Vision science: Photons to phenomenology. MIT Press.
- Rock, I. (1983). The logic of perception. MIT Press.
