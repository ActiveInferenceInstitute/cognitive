---
title: Dark Adaptation
description: The recovery of visual sensitivity during time spent in low light, with its two-phase course reflecting cone and rod mechanisms and the regeneration of photopigment.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - vision
  - adaptation
  - psychophysics
semantic_relations:
  - type: relates
    links:
      - "[[light_adaptation]]"
      - "[[brightness_perception]]"
      - "[[sensory_adaptation]]"
      - "[[contrast_sensitivity]]"
      - "[[gain_control]]"
wikidata: Q22273766
---

# Dark Adaptation

## Overview
Dark adaptation is the gradual increase in the eye's sensitivity after a move from bright light to darkness. A person entering a dim room first sees little, but sensitivity recovers over many minutes and the absolute threshold for detecting light falls by several orders of magnitude. The reverse process, adjustment to bright light, is light adaptation and is much faster.

## Time course
Measured by determining the dimmest detectable test flash at intervals in darkness, the threshold curve has two limbs. The first, rapid limb reflects cones and levels off after several minutes. The second reflects rods, which are slower but ultimately far more sensitive, and the curve begins to fall again at the rod-cone break and continues to improve for tens of minutes. A test flash that is red-biased or confined to the fovea, where rods are absent, follows only the cone limb. Rods are concentrated in the periphery, so dim stimuli are best seen when looking slightly away from them.

## Mechanisms
- **Photopigment regeneration.** Bright light bleaches the visual pigments, rhodopsin in rods and cone opsins in cones. Sensitivity recovers as the retinal chromophore is regenerated through the visual cycle, which is slower for rods.
- **Neural adaptation.** Gain changes in the photoreceptors, through calcium-dependent feedback in the phototransduction cascade, and in downstream retinal circuits adjust sensitivity independently of the pigment level.
- **Pooling.** Rod signals converge onto bipolar and ganglion cells, which raises sensitivity at the cost of spatial acuity.

## Relation to Active Inference
Dark adaptation is a sensory mechanism and has no dedicated Active Inference theory. It can be viewed as an example of gain control that keeps sensory signals within a useful range as the statistics of the input change. In predictive-processing terms this corresponds to adjusting the expected precision of sensory evidence: in the dark, signals are weaker and noisier, so low-level evidence is weighted less. This remains an interpretive analogy.

## Related concepts
- [[light_adaptation]] — the converse, rapid adjustment to bright light
- [[brightness_perception]] — how adaptation state changes apparent brightness
- [[sensory_adaptation]] — the general class of adjustments to stimulus conditions
- [[contrast_sensitivity]] — which declines at low light levels
- [[gain_control]] — the computational principle behind adaptation
- [[peripheral_vision]] — where rod-based vision dominates

## References
- Kandel, E. R., Schwartz, J. H., Jessell, T. M., Siegelbaum, S. A., & Hudspeth, A. J. (Eds.). (2013). Principles of Neural Science (5th ed.). McGraw-Hill.
- Lamb, T. D., & Pugh, E. N., Jr. (2004). Dark adaptation and the retinoid cycle of vision. Progress in Retinal and Eye Research, 23, 307-380.
