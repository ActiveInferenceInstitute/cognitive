---
title: Shape Constancy
description: The perceptual stability of an object's shape across changes in viewing angle, covering the geometry of projection, the cues involved, and an inferential reading within predictive processing.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - perception
  - vision
  - form
semantic_relations:
  - type: relates
    links:
      - "[[perceptual_constancy]]"
      - "[[size_constancy]]"
      - "[[form_perception]]"
      - "[[perceptual_inference]]"
---

# Shape Constancy

## Overview
Shape constancy is the tendency to perceive an object as having the same shape when it is seen from different angles. A coin tilted away from the observer projects an ellipse onto the retina, yet it continues to look circular. A rectangular table viewed obliquely is seen as rectangular, not as a trapezoid. Shape constancy is one form of [[perceptual_constancy]] and, like the others, depends on information about the viewing conditions, here the orientation of the surface relative to the observer.

## Geometry and cues
Under orthographic or perspective projection, the image of a planar shape is a foreshortened version of that shape. If the slant of the surface is known, the foreshortening can be inverted and the true shape recovered. The visual system appears to combine several sources of slant information:
- **Texture gradients**, in which the density and shape of surface elements change systematically with slant.
- **Binocular disparity** and **motion parallax**, which give direct information about depth variation across a surface.
- **Perspective cues**, such as converging edges, and the context provided by surrounding surfaces.
- **Familiarity and symmetry**, which supply priors about what shapes are likely (see [[object_recognition]]).

Experimental work indicates that constancy is usually partial. Observers' judgements typically lie between the true shape and the shape of the retinal projection, and the compromise moves toward the true shape as depth cues become richer. Phenomenal reports also differ from judgements made under instructions to match the projected outline, which suggests that both the distal and proximal descriptions are available to the observer.

## Relation to Active Inference
The inferential account treats shape constancy as a joint estimate of shape and orientation from a single image. A generative model specifies how three-dimensional shapes and slants produce images, and the percept is the posterior over its latent variables (see [[perceptual_inference]], [[generative_model]]). Priors such as a preference for symmetric or rectangular shapes resolve ambiguity when cues are weak, and the combination of cues can be treated as precision-weighted evidence (see [[precision_weighting]]). The link to Active Inference is therefore an application of general Bayesian perceptual principles, not a distinct derivation. Active sampling is also relevant, since moving to view an object from another angle produces new evidence that constrains its shape.

## Related concepts
- [[perceptual_constancy]] — the general phenomenon of stable percepts
- [[size_constancy]] — the parallel case for perceived size
- [[form_perception]] — how shapes and contours are organised
- [[depth_perception]] — the cues that supply slant information
- [[perceptual_inference]] — the inferential framework

## References
- Palmer, S. E. (1999). Vision science: Photons to phenomenology. MIT Press.
- Walsh, V., & Kulikowski, J. (Eds.). (1998). Perceptual constancy: Why things look as they do. Cambridge University Press.
- Rock, I. (1983). The logic of perception. MIT Press.
