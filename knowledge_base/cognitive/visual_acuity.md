---
title: Visual Acuity
description: "The eye's ability to resolve fine spatial detail, its optical and neural limits, measurement by letter charts and gratings, decline with eccentricity, and acuity as precision of sensory evidence."
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - vision
  - psychophysics
  - retina
semantic_relations:
  - type: relates
    links:
      - "[[foveal_vision]]"
      - "[[contrast_sensitivity]]"
      - "[[peripheral_vision]]"
      - "[[psychophysics]]"
wikidata: Q555066
---

# Visual Acuity

## Overview
Visual acuity is the smallest spatial detail an observer can resolve. It is expressed as the minimum angle of resolution (MAR), the smallest angular separation between two features that can be distinguished, or as the reciprocal measure of spatial frequency in cycles per degree. Normal acuity of "20/20" corresponds to a MAR of one minute of arc.

## Measurement and limits
- **Letter charts.** Snellen-type charts use optotypes whose strokes subtend a set angle; a letter at the 20/20 line is five arcminutes tall with strokes of one arcminute. LogMAR charts use equal logarithmic steps. A MAR of 1 arcminute corresponds to a grating of 30 cycles per degree, since one cycle contains a light and a dark bar.
- **Types.** Detection (is something there), resolution (are two features separate), recognition (what is it), and hyperacuity, such as vernier alignment, where thresholds are far smaller than the spacing between photoreceptors. Hyperacuity shows that acuity is not limited by receptor spacing alone.
- **Limiting factors.** Optical blur from the eye's optics, cone spacing and density in the fovea, and neural pooling downstream (retinal ganglion cells and cortex). Luminance, contrast, pupil size, and eye movements also matter.
- **Eccentricity.** Acuity is highest in the fovea and falls steeply toward the periphery as cone density declines and receptive fields grow ([[foveal_vision]], [[peripheral_vision]]).
- **Relation to contrast sensitivity.** Acuity is the high-frequency cut-off of the contrast sensitivity function; two observers with equal acuity can differ in sensitivity at lower frequencies ([[contrast_sensitivity]]).

## Role in Active Inference
Acuity can be read as the precision of the sensory likelihood in a given part of the visual field: foveal observations are high-precision and peripheral ones low-precision. Eye movements then become actions that place high-precision sensory sampling where it resolves the most uncertainty, as in the "saccades as experiments" account of Friston and colleagues ([[epistemic_foraging]], [[precision_weighting]]). This is a modelling interpretation; the optical and retinal limits listed above are independent of it.

## Related concepts
- [[foveal_vision]] - the region of highest acuity
- [[peripheral_vision]] - lower acuity and wider coverage
- [[contrast_sensitivity]] - the related spatial-frequency measure
- [[spatial_attention]] - which can modulate effective resolution
- [[visual_perception]] - the wider field

## References
- Friston, K., Adams, R. A., Perrinet, L., & Breakspear, M. (2012). Perceptions as hypotheses: saccades as experiments. Frontiers in Psychology.
