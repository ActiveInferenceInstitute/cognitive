---
title: Overt Attention
description: Attention that is shifted by moving the eyes, head or body toward a target, its relation to covert attention and saliency, and its treatment as active sampling in Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - attention
  - eye_movements
  - active_sensing
semantic_relations:
  - type: relates
    links:
      - "[[covert_attention]]"
      - "[[eye_movements]]"
      - "[[visual_attention]]"
      - "[[frontal_eye_fields]]"
      - "[[active_sensing]]"
---

# Overt Attention

## Overview
Overt attention is the selection of information by orienting the sensory apparatus, most often by moving the eyes so that a region of interest falls on the fovea, where acuity is highest. It is contrasted with covert attention, in which processing is prioritised for a location while gaze stays fixed. The two forms are closely coupled, since an upcoming saccade is preceded by a shift of covert attention toward its goal, a link often called the premotor theory of attention.

## Mechanism
Vision is sampled in a sequence of fixations of a few hundred milliseconds separated by saccades, during which little useful visual information is acquired. Where the eyes go next depends on both bottom-up and top-down factors:
- **Stimulus-driven**: locations that differ from their surroundings in luminance, colour or motion attract gaze, a property captured by saliency models (Itti and Koch, 2001).
- **Goal-driven**: task demands, knowledge of scene structure and reward strongly shape fixation patterns, and in many everyday tasks they outweigh low-level saliency.
- **History-dependent**: mechanisms such as inhibition of return reduce the tendency to look again at recently visited locations.

The control of saccades involves the superior colliculus, the frontal eye fields ([[frontal_eye_fields]]), parietal cortex and brainstem circuits. Combined maps of salience and relevance are often described as priority maps ([[priority_maps]]).

## Measurement
Eye trackers record fixation locations, durations and saccade amplitudes. Free-viewing and visual search ([[visual_search]]) tasks are widely used. Fixation records give an observable index of attention that covert attention lacks, although a fixated location is not always the one being processed.

## Relation to Active Inference
The connection is a developed modelling programme. Friston and colleagues (2012) proposed that saccades are experiments chosen to test hypotheses about the scene, which in the Active Inference formalism corresponds to selecting fixations that minimise expected free energy, and in particular maximise expected information gain about hidden causes of sensory input. Simulations with this scheme reproduce features of free viewing, such as scan paths directed to informative features and inhibition of return. Such models show how epistemic value can explain gaze selection, but they are demonstrations on simplified stimuli, and their quantitative match to human fixation data is still being evaluated.

## Related concepts
- [[covert_attention]] — attention shifted without movement of the eyes
- [[eye_movements]] — the motor behaviours that implement overt orienting
- [[visual_attention]] — the broader study of selection in vision
- [[active_sensing]] — choosing actions to acquire informative data
- [[priority_maps]] — spatial maps that guide the next fixation

## References
- Findlay, J. M., & Gilchrist, I. D. (2003). Active Vision: The Psychology of Looking and Seeing. Oxford University Press.
- Itti, L., & Koch, C. (2001). Computational modelling of visual attention. Nature Reviews Neuroscience, 2, 194-203.
- Friston, K., Adams, R. A., Perrinet, L., & Breakspear, M. (2012). Perceptions as hypotheses: Saccades as experiments. Frontiers in Psychology, 3, 151.
