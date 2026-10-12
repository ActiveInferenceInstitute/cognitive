---
title: Pursuit Movements
description: Smooth pursuit eye movements that keep the image of a moving target near the fovea, their neural pathways, their dependence on prediction, and Active Inference models of oculomotor control.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - eye_movements
  - oculomotor_control
  - motion_perception
semantic_relations:
  - type: relates
    links:
      - "[[eye_movements]]"
      - "[[motion_perception]]"
      - "[[middle_temporal]]"
      - "[[frontal_eye_fields]]"
      - "[[cerebellum]]"
      - "[[efference_copy]]"
---

# Pursuit Movements

## Overview
Smooth pursuit is the class of slow, continuous eye movements by which primates keep the image of a moving object near the fovea, where acuity is highest. It differs from saccades, which are fast jumps of gaze, and from the vestibulo-ocular reflex, which stabilises gaze during head movement ([[eye_movements]]).

## Characteristics
When a target begins to move unpredictably, the eyes start to follow after a delay of roughly a hundred milliseconds. The first part of the response, the initiation, is driven by the visual motion of the target across the retina without feedback from the eye's own movement. After this open-loop phase, the eyes match target velocity and maintenance is controlled with feedback. If eye velocity falls behind the target, small catch-up saccades correct the position error.

## Neural pathways
Motion signals are analysed in the middle temporal area and the adjacent medial superior temporal area ([[middle_temporal]], [[motion_perception]]), which project to the frontal eye fields ([[frontal_eye_fields]]) and other cortical regions and then to the pontine nuclei and the cerebellum. The cerebellar flocculus and the vermis help to adjust the gain and timing of pursuit, and brainstem circuits produce the motor commands (Lisberger, 2010). Krauzlis (2004) proposed that pursuit shares target selection mechanisms with saccades, with the superior colliculus and frontal eye fields involved in both.

## Prediction
Pursuit is not a purely reactive tracking process. Eye movements can anticipate the onset of a target with a known direction, and they continue for a short time when the target disappears behind an occluder, using an internal estimate of target velocity that may depend on a copy of the motor command ([[efference_copy]]). Pursuit is commonly impaired in schizophrenia and in cerebellar and frontal disorders, and measures of pursuit have been studied as markers of these conditions.

## Relation to Active Inference
The relation is a modelling programme. Adams, Perrinet and Friston (2012) simulated pursuit with and without occlusion in an Active Inference scheme in which the eyes move to fulfil proprioceptive predictions about the target's position. The model reproduces continued tracking during occlusion when beliefs about the target's trajectory are held with sufficient precision. They suggested that reduced precision of such beliefs might account for the pursuit deficit in schizophrenia. This is a hypothesis from simulation, and its clinical application requires empirical testing.

## Related concepts
- [[eye_movements]] — the family of gaze movements, including saccades and pursuit
- [[motion_perception]] — analysis of target motion that drives pursuit
- [[middle_temporal]] — cortical area representing visual motion
- [[cerebellum]] — structure that calibrates the gain of pursuit
- [[efference_copy]] — internal signals used to maintain tracking during occlusion

## References
- Lisberger, S. G. (2010). Visual guidance of smooth-pursuit eye movements: Sensation, action, and what happens in between. Neuron, 66, 477-491.
- Krauzlis, R. J. (2004). Recasting the smooth pursuit eye movement system. Journal of Neurophysiology, 91, 591-603.
- Adams, R. A., Perrinet, L. U., & Friston, K. (2012). Smooth pursuit and visual occlusion: Active inference and oculomotor control in schizophrenia. PLoS ONE, 7, e47502.
