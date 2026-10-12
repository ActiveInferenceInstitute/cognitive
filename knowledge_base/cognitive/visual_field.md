---
title: Visual Field
description: The region of space visible at a given moment with the eyes fixed, its extent, retinotopic mapping to the visual cortex, common field defects, and its role in attention and perception.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - vision
  - retinotopy
  - neuroanatomy
semantic_relations:
  - type: relates
    links:
      - "[[peripheral_vision]]"
      - "[[visual_acuity]]"
      - "[[visual_cortex]]"
      - "[[visual_attention]]"
      - "[[retinal_implants]]"
---

# Visual Field

## Overview
The visual field is the portion of space that can be seen at one time while the eyes and head are held still. With both eyes open, the field extends roughly 200 degrees horizontally, with a central region of about 120 degrees seen by both eyes, which supports binocular depth perception. The remaining lateral regions are seen by one eye only. 

## Organisation and mapping
Each eye's field is divided into nasal and temporal halves, and each half into upper and lower quadrants. Light from the left half of the visual field of both eyes falls on the right half of each retina, and the optic nerve fibres from the nasal retinae cross at the optic chiasm. As a result, the left visual field is represented in the right cerebral hemisphere and the right field in the left, through the lateral geniculate nucleus and optic radiations to primary visual cortex. The representation is retinotopic: adjacent points in the field map to adjacent points in cortex, in the upper field below the calcarine fissure and the lower field above it. Further cortical areas contain their own field maps (Wandell, Dumoulin and Brewer, 2007).

The map is non-uniform. The fovea, a small region at the centre of the field, occupies a disproportionately large share of primary visual cortex, an effect known as cortical magnification. Acuity falls steeply with eccentricity (see [[visual_acuity]], [[peripheral_vision]]), so the periphery contributes mainly to detecting motion and guiding attention and eye movements. A normal blind spot lies about 15 degrees temporal to fixation, where the optic nerve leaves the retina.

## Defects
Damage along the pathway produces characteristic defects whose pattern indicates the location of the lesion. A scotoma is a localised area of reduced or absent vision, and a hemianopia is the loss of half of the field. A defect on the same side of the field in both eyes points to damage behind the optic chiasm. Perimetry maps sensitivity across the field.

## Relation to Active Inference
The visual field is the sensory space over which a generative model must predict inputs. Active Inference treats eye movements as actions that move the high-resolution fovea to locations expected to resolve the most uncertainty about the scene, in simulations of visual search and saccades. The link is a research programme on active sensing, not a settled account. See [[visual_attention]].

## Related concepts
- [[peripheral_vision]] — vision outside the central field
- [[visual_acuity]] — resolution that varies across the field
- [[visual_cortex]] — cortical maps of the field
- [[visual_attention]] — selection within the field
- [[retinal_implants]] — prostheses that stimulate field regions
- [[eye_movements]] — actions that sample the field

## References
- Hubel, D. H., & Wiesel, T. N. (1968). Receptive fields and functional architecture of monkey striate cortex. Journal of Physiology, 195, 215-243.
- Wandell, B. A., Dumoulin, S. O., & Brewer, A. A. (2007). Visual field maps in human cortex. Neuron, 56, 366-383.
- Kandel, E. R., Schwartz, J. H., Jessell, T. M., Siegelbaum, S. A., & Hudspeth, A. J. (Eds.). (2013). Principles of neural science (5th ed.). McGraw-Hill.
