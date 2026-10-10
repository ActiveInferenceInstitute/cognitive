---
title: Action Recognition
description: "How observers identify and understand actions performed by others, covering biological motion, the action observation network, mirror neurons and their debated role, and a predictive-coding account."
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - social-cognition
  - perception
  - mirror-neurons
semantic_relations:
  - type: relates
    links:
      - "[[biological_motion]]"
      - "[[motion_perception]]"
      - "[[social_cognition]]"
---

# Action Recognition

## Overview
Action recognition is the identification of what another agent is doing, and often why, from observation. It ranges from labelling a movement (reaching, grasping) to inferring a goal or intention. This page concerns the cognitive and neural sense; automated video classification of human actions is a computer-vision task that shares the name but not the mechanisms.

## Mechanism
Perception of body movement is remarkably efficient. Johansson showed in 1973 that a few lights attached to the joints of a moving person are enough for observers to perceive walking and other actions, which established biological motion as a distinct perceptual category ([[biological_motion]], [[motion_perception]]). The posterior superior temporal sulcus is a key region for analysing such movement, and it connects to parietal and premotor areas that together form an action observation network, including the [[inferior_parietal]] cortex.

Mirror neurons were discovered by single-unit recording ([[single_unit]]) in monkey premotor area F5 and parietal cortex: some cells fire both when the animal performs a grasp and when it watches someone else do so. This led to the proposal that action understanding works by mapping observed actions onto the observer's own motor repertoire ([[imitation]]). The strong form of this claim is contested. Critics such as Hickok argue that monkey mirror neuron findings do not show understanding, that action understanding survives damage to motor systems in humans, and that mirror responses may reflect prediction or association rather than comprehension. Evidence in humans comes from imaging, which cannot isolate single neurons, and a more cautious view treats the motor system as contributing to prediction of observed actions rather than being necessary for recognition.

Recognition of intention depends on context: the same grasp can be for eating or for placing, and observers use object, setting and prior expectation ([[theory_of_mind]], [[perspective_taking]]). Infants show sensitivity to goal-directed actions early ([[cognitive_development]]), and learning by watching others builds on this capacity ([[observational_learning]]).

## Role in Active Inference
Kilner, Friston and Frith proposed that the mirror system is best understood as hierarchical predictive coding. Observed movement is explained by inferring the intention that would generate it under the observer's own model of action, with prediction errors propagating through the levels from kinematics to goals ([[predictive_coding]], [[generative_model]]). This account makes action understanding an inference problem and predicts that expectations modulate mirror responses; it is a theoretical framework with partial empirical support and remains debated alongside alternatives.

## Related concepts
- [[motor_control]] — the production side of the action repertoire
- [[social_neuroscience]] — brain systems for understanding others
- [[multi_agent_active_inference]] — modelling agents that infer each other's goals

## References
- Johansson, G. (1973). Visual perception of biological motion and a model for its analysis. *Perception & Psychophysics*, 14.
- Kilner, J. M., Friston, K. J., & Frith, C. D. (2007). Predictive coding: an account of the mirror neuron system. *Cognitive Processing*, 8.
- Hickok, G. (2009). Eight problems for the mirror neuron theory of action understanding in monkeys and humans. *Journal of Cognitive Neuroscience*, 21.
