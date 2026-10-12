---
title: Developmental Robotics
description: The field that uses robots to model how cognitive abilities develop through embodied interaction over time, its principles such as intrinsic motivation, and its overlap with Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - robotics
  - cognitive_development
  - embodiment
semantic_relations:
  - type: relates
    links:
      - "[[cognitive_robotics]]"
      - "[[robotics]]"
      - "[[sensorimotor_development]]"
      - "[[cognitive_development]]"
      - "[[embodied_cognition]]"
      - "[[active_learning]]"
wikidata: Q5266817
---

# Developmental Robotics

## Overview
Developmental robotics, also called epigenetic robotics, is an interdisciplinary approach in which robots are built to acquire skills gradually through interaction with their environment and with people, instead of being fully programmed in advance. It has two aims: to produce more adaptable machines, and to use robots as synthetic models for testing theories from developmental psychology and neuroscience (Lungarella et al., 2003; Asada et al., 2009).

## Principles
- **Embodiment.** Cognition is treated as emerging from the coupling of body, brain and environment, so the robot's morphology and sensors shape what it learns (see [[embodied_cognition]]).
- **Staged, open-ended learning.** Capabilities build on earlier ones, inspired by accounts of infant sensorimotor development such as the progression from reflexes to reaching, grasping and tool use.
- **Intrinsic motivation.** Robots are driven by internal signals such as novelty, prediction error or learning progress, instead of only external rewards (Oudeyer & Kaplan, 2007).
- **Social interaction.** Caregiver-like guidance, imitation and joint attention are used to scaffold learning.

Child-sized humanoid platforms such as iCub were designed to support research of this kind (Cangelosi & Schlesinger, 2015). Typical studies model reaching, body schema learning, object manipulation and early word learning grounded in perception and action.

## Methodological issues
Robot models are simplified, and success in reproducing a behaviour does not show that infants use the same mechanism. The value of the approach lies in making assumptions explicit and in generating predictions that can be tested with children.

## Relation to Active Inference
The overlap is substantial. Curiosity and learning progress in developmental robotics are conceptually related to the epistemic, information-seeking term of expected free energy, and Active Inference provides a single objective for perception, learning and action in an embodied agent (see [[active_learning]]). Active Inference controllers have been implemented on robots as an active research programme, though few address long-term developmental trajectories, and the framework is one of several used in the field.

## Related concepts
- [[cognitive_robotics]] — the broader field of robots with cognitive abilities
- [[robotics]] — the engineering context
- [[sensorimotor_development]] — the infant processes being modelled
- [[cognitive_development]] — the psychological theories that inspire it
- [[embodied_cognition]] — the theoretical stance on body and mind

## References
- Lungarella, M., Metta, G., Pfeifer, R., & Sandini, G. (2003). Developmental robotics: A survey. Connection Science, 15, 151-190.
- Asada, M., Hosoda, K., Kuniyoshi, Y., Ishiguro, H., Inui, T., Yoshikawa, Y., Ogino, M., & Yoshida, C. (2009). Cognitive developmental robotics: A survey. IEEE Transactions on Autonomous Mental Development, 1, 12-34.
- Oudeyer, P.-Y., & Kaplan, F. (2007). What is intrinsic motivation? A typology of computational approaches. Frontiers in Neurorobotics, 1, 6.
- Cangelosi, A., & Schlesinger, M. (2015). Developmental Robotics: From Babies to Robots. MIT Press.
