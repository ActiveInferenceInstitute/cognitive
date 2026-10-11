---
title: Color Constancy
description: The tendency of surfaces to keep a stable perceived colour despite changes in illumination, the mechanisms that support it, its limits, and its interpretation as Bayesian inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - vision
  - perceptual_constancy
  - bayesian_perception
semantic_relations:
  - type: relates
    links:
      - "[[perceptual_constancy]]"
      - "[[color_processing]]"
      - "[[perceptual_inference]]"
      - "[[visual_cortex]]"
      - "[[light_adaptation]]"
---

# Color Constancy

## Overview
Color constancy is the tendency of a surface to keep a roughly stable perceived colour when the spectrum of the light falling on it changes. The light reaching the eye is the product of the illuminant and the surface reflectance, so a single set of cone signals is consistent with many combinations of the two. A banana looks yellow at noon and at dusk even though the light it reflects differs substantially. Constancy is imperfect, and the degree achieved varies with the scene and the task.

## Mechanisms
Several processes contribute. Adaptation of the cone photoreceptors scales each cone class's signal by the prevailing light level, in the manner described as von Kries scaling. Comparisons between neighbouring surfaces remove much of the illuminant, an idea developed in Land's retinex theory, in which lightness is computed from ratios of signals across edges (Land and McCann, 1971). Other cues include the average colour of the scene, specular highlights, shadows and learned expectations about familiar objects. Neurons sensitive to colour contrast in primary visual cortex and responses in later ventral areas, including V4, have been linked to constancy, though the neural locus remains debated. Foster (2011) reviews the evidence.

## Limits and individual differences
Constancy fails under unusual lighting or when cues are removed. A photograph of a dress that circulated in 2015 was seen as blue and black by some observers and white and gold by others, a difference attributed to different assumptions about the illuminant (Brainard and Hurlbert, 2015).

## Relation to Active Inference
Color constancy is a textbook example of perception as inverse inference: the visual system estimates hidden causes (surface reflectance and illuminant) from ambiguous data using prior knowledge about likely lights and surfaces. Bayesian models fitted to behavioural data capture many results (Brainard and Freeman, 1997). Within the predictive-processing family this is a straightforward instance of [[perceptual_inference]]; it does not test anything specific to Active Inference, because no action or free-energy-specific prediction is involved. The individual differences seen with the dress image are often explained as different priors.

## Related concepts
- [[perceptual_constancy]] — the general class of stabilised percepts
- [[color_processing]] — the early stages of chromatic vision
- [[perceptual_inference]] — the inferential reading of perception
- [[visual_cortex]] — the cortical areas implicated in colour
- [[light_adaptation]] — receptor adjustment that underlies part of constancy

## References
- Land, E. H., & McCann, J. J. (1971). Lightness and retinex theory. Journal of the Optical Society of America, 61, 1-11.
- Brainard, D. H., & Freeman, W. T. (1997). Bayesian color constancy. Journal of the Optical Society of America A, 14, 1393-1411.
- Foster, D. H. (2011). Color constancy. Vision Research, 51, 674-700.
- Brainard, D. H., & Hurlbert, A. C. (2015). Colour vision: understanding #TheDress. Current Biology, 25, R551-R554.
