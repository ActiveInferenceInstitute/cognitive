---
title: Motor Cortex
description: The frontal-lobe cortical areas, primary motor, premotor and supplementary motor, that plan and issue voluntary movement commands, and how Active Inference reinterprets motor output as proprioceptive prediction.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - neuroanatomy
  - motor_control
  - action
semantic_relations:
  - type: relates
    links:
      - "[[motor_control]]"
      - "[[motor_learning]]"
      - "[[cerebellum]]"
      - "[[action_selection]]"
      - "[[sensorimotor_coordination]]"
wikidata: Q18696
---

# Motor Cortex

## Overview
The motor cortex comprises the regions of the frontal lobe that contribute to planning, initiating and controlling voluntary movement. The primary motor cortex (M1) lies in the precentral gyrus, Brodmann area 4, and contains the large pyramidal Betz cells whose axons form part of the corticospinal tract. Anterior to it lie the premotor cortex and the supplementary motor area, which are more involved in selecting and sequencing movements, and in guiding action by external cues or internal plans.

## Organisation and coding
M1 has a somatotopic map, often drawn as a motor homunculus, established by electrical stimulation of the human cortex (Penfield & Boldrey, 1937). Regions that control fine movement, such as the hand and face, occupy a disproportionately large area. The map is not a strict point-to-point layout: neighbouring body parts overlap, and the representation can reorganise after injury or training. Recordings in monkeys showed that single M1 neurons are broadly tuned to the direction of reaching, and that the direction is accurately predicted by the combined activity of a population (Georgopoulos et al., 1986). Whether M1 activity is best described in terms of muscle forces, movement kinematics or a dynamical system that evolves over time remains debated. Lesions or stroke affecting the cortex or its descending pathway produce weakness or paralysis on the opposite side of the body (see [[motor_control]]).

## Relation to Active Inference
Active Inference offers a distinctive reading of motor cortex. Rather than sending commands, descending signals are proposed to be predictions of proprioceptive input. Spinal reflex arcs then move the muscles until the sensed state matches the prediction, which resolves prediction error through action (Adams et al., 2013). Shipp, Adams and Friston (2013) argued that the agranular structure of motor cortex, which lacks the granular input layer of sensory cortex, fits this role, as it would be specialised for generating predictions and receiving fewer ascending errors. This is a leading theoretical programme with anatomical support, but it competes with optimal feedback control accounts, and the data do not yet decisively favour either. See [[action_selection]] for how policies are chosen.

## Related concepts
- [[motor_control]] — the computational problem addressed by the motor system
- [[motor_learning]] — plasticity in M1 and premotor areas with practice
- [[cerebellum]] — structure working with motor cortex to refine movement
- [[action_selection]] — choosing among possible movements
- [[sensorimotor_coordination]] — integration of sensory feedback with movement

## References
- Penfield, W., & Boldrey, E. (1937). Somatic motor and sensory representation in the cerebral cortex of man as studied by electrical stimulation. Brain, 60, 389-443.
- Georgopoulos, A. P., Schwartz, A. B., & Kettner, R. E. (1986). Neuronal population coding of movement direction. Science, 233, 1416-1419.
- Adams, R. A., Shipp, S., & Friston, K. J. (2013). Predictions not commands: active inference in the motor system. Brain Structure and Function, 218, 611-643.
- Shipp, S., Adams, R. A., & Friston, K. J. (2013). Reflections on agranular architecture: predictive coding in the motor cortex. Trends in Neurosciences, 36, 706-716.
