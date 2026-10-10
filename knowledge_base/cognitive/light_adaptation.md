---
title: Light Adaptation
description: Light adaptation is the visual system's adjustment of sensitivity to ambient luminance, covering photoreceptor and retinal mechanisms, Weber behaviour, and its relation to gain control and precision.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - vision
  - adaptation
  - retina
semantic_relations:
  - type: relates
    links:
      - "[[brightness_perception]]"
      - "[[gain_control]]"
      - "[[visual_perception]]"
---

# Light Adaptation

## Overview
Light adaptation is the process by which the visual system lowers its sensitivity as ambient luminance increases, so that it can signal contrast across a range of illumination spanning many orders of magnitude (very roughly nine or ten log units between starlight and bright sun) with neurons that have a limited response range. Its counterpart, dark adaptation, is the slower recovery of sensitivity after moving to low light. The result is that perceived brightness and contrast depend on local luminance relative to the adapted state, not absolute intensity ([[brightness_perception]]).

## Mechanism
- **Weber behaviour**: at moderate to high luminance, the smallest detectable increment $\Delta I$ is approximately proportional to background intensity $I$, so $\Delta I / I$ is roughly constant, on the order of a percent or two under favourable conditions. At very low light, sensitivity is limited by photon noise instead.
- **Photoreceptor feedback**: calcium entering through open channels in the outer segment regulates phototransduction through guanylate cyclase activation and receptor recovery, shortening and desensitizing responses as light rises.
- **Bleaching**: at high intensities, photopigment is bleached, reducing quantum catch and slowing recovery; this dominates the long time course of dark adaptation.
- **Duplex vision**: rods dominate at low (scotopic) luminance and cones at high (photopic) luminance, giving a two-phase dark adaptation curve with a rod-cone break after several minutes.
- **Retinal circuits**: horizontal and bipolar cell interactions provide spatial and temporal gain control beyond the receptors.
- **Pupil**: pupil diameter varying from about 2 to 8 mm changes retinal illuminance by at most a factor of about 16, a minor contributor compared with the neural mechanisms.

## Role in Active Inference
Adaptation is usually interpreted through efficient coding: neural responses are rescaled to the statistics of the current input so that limited dynamic range encodes the relevant variation. Within predictive-processing accounts, this resembles a change in the expected mean and variance of sensory input, with the adapted state acting as a prior and responses encoding deviations from it. Rescaling response gain is also the computational role assigned to precision ([[precision_weighting]]). The correspondence is an interpretation: photoreceptor biochemistry is well described without invoking free-energy minimization, and Active Inference adds a normative reading, not a different mechanism.

## Related concepts
- [[brightness_perception]] — perceptual consequences of the adapted state
- [[gain_control]] — general principle of rescaling responsiveness
- [[visual_perception]] — wider processing context
- [[contrast_sensitivity]] — sensitivity measure that adaptation shifts
- [[adaptation_mechanisms]] — adaptation across sensory and cognitive systems
- [[psychophysics]] — methods used to measure thresholds and Weber fractions
