---
title: Eye Movements
description: The types of eye movement, including saccades, smooth pursuit and vergence, their neural control, role in visual sampling, and the Active Inference view of saccades as experiments.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - vision
  - motor_control
  - active_sensing
semantic_relations:
  - type: relates
    links:
      - "[[visual_attention]]"
      - "[[frontal_eye_fields]]"
      - "[[binocular_vision]]"
      - "[[epistemic_foraging]]"
      - "[[visual_perception]]"
---

# Eye Movements

## Overview
Eye movements reposition the high-acuity fovea, stabilise images on the retina and align the two eyes. Because only a small central region of the retina sees in fine detail, vision depends on sampling the scene with a sequence of movements and fixations rather than on a single static image. Their study links perception, attention and motor control.

## Main types
- **Saccades**: rapid, ballistic shifts of gaze, lasting a few tens of milliseconds, which occur several times a second during viewing and reading. Perception is strongly suppressed during them.
- **Fixations**: periods of relative stillness, typically a few hundred milliseconds, during which visual information is acquired.
- **Smooth pursuit**: slow tracking of a moving target.
- **Vestibulo-ocular and optokinetic reflexes**: compensatory movements that keep the image stable during head or scene motion.
- **Vergence**: opposite-direction movements that adjust the eyes to viewing distance (see [[binocular_vision]]).
- **Microsaccades**: small involuntary movements during fixation.

Control involves the superior colliculus, brainstem gaze centres, the cerebellum, and cortical areas including the [[frontal_eye_fields]] and parietal cortex, which link eye movements to [[visual_attention]]. Saccades are closely coupled to shifts of covert attention to the target location before the eyes move.

Where people look is not random. Gaze patterns depend on stimulus properties such as contrast, on the observer's task, and on prior knowledge: Yarbus (1967) famously showed that the same picture is scanned differently under different instructions. In reading, fixation durations and skips vary with word frequency and predictability (Rayner, 1998).

## Relation to Active Inference
The link is strong and an active research programme. In Active Inference, an agent samples the world to reduce uncertainty about hidden states, and saccades are the clearest example. Friston and colleagues (2012) described saccades as experiments the brain performs to test hypotheses about a scene: gaze is directed to locations where the expected information gain is greatest, which corresponds to the epistemic term of expected free energy. Simulations of this scheme reproduce features of scan paths, including inhibition of return. The account makes gaze selection a form of policy selection over where to look, closely related to [[epistemic_foraging]] and [[information_gain]]. Empirical tests against human eye-tracking data are still developing, and other models of saliency and reward also explain much of gaze behaviour.

## Related concepts
- [[visual_attention]] — covert and overt selection of visual input
- [[frontal_eye_fields]] — cortical control of voluntary saccades
- [[binocular_vision]] — vergence and depth from two eyes
- [[epistemic_foraging]] — information-seeking action
- [[information_gain]] — the quantity gaze may maximise

## References
- Yarbus, A. L. (1967). Eye Movements and Vision. Plenum Press.
- Rayner, K. (1998). Eye movements in reading and information processing: 20 years of research. Psychological Bulletin, 124, 372-422.
- Findlay, J. M., & Gilchrist, I. D. (2003). Active Vision: The Psychology of Looking and Seeing. Oxford University Press.
- Friston, K., Adams, R. A., Perrinet, L., & Breakspear, M. (2012). Perceptions as hypotheses: saccades as experiments. Frontiers in Psychology, 3, 151.
