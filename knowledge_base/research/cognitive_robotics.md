---
title: Cognitive Robotics
type: knowledge_base
status: stable
created: 2026-02-07
updated: 2026-10-09
tags:
  - research
  - active_inference
semantic_relations:
  - type: relates_to
    links:
      - "[[knowledge_base/research/robot_control]]"
      - "[[knowledge_base/cognitive/active_inference]]"
      - "[[knowledge_base/research/social_cognition]]"
      - "[[knowledge_base/free_energy_principle/implementations/robotics]]"
      - "[[knowledge_base/cognitive/embodied_cognition]]"
      - "[[knowledge_base/cognitive/enactive_cognition]]"
      - "[[knowledge_base/cognitive/sensorimotor_coordination]]"
      - "[[knowledge_base/cognitive/epistemic_foraging]]"
---

# Cognitive Robotics

## Overview

Cognitive robotics studies robots that perceive, learn, reason and act with some of the flexibility of animals. It draws on embodied cognition, developmental robotics (where a robot's skills grow through interaction, as with the iCub humanoid), and human-robot interaction. The field's premise is that intelligence depends on having a body that senses and acts, so modelling cognition benefits from building it into hardware.

## Main questions

- **Body representation.** How a robot learns a body schema, meaning the relation between joint configuration, vision and touch, and keeps it calibrated after damage or tool use.
- **Sensorimotor learning.** How skills are acquired from interaction, including exploration driven by curiosity.
- **Planning and decision-making** under uncertainty, in tasks that mix goals at several time scales.
- **Interaction.** How a robot infers a person's intentions and communicates its own, and how it can behave predictably in shared spaces.

## Connection to Active Inference

The connection is an active research programme, with demonstrations that are mostly in simulation or on single research platforms. The framework fits the field's premises because it treats perception and action as one process driven by a generative model of the body and world. Concretely, body schema learning is the update of parameters of the model that links proprioception, vision and touch; reaching is the fulfilment of a predicted limb position; and curiosity is the epistemic term of expected free energy. Lanillos and colleagues' survey, *Active inference in robotics and artificial agents: survey and challenges* (2021), gives an overview and is candid that most work addresses small problems. Interaction is typically cast as inference about another agent's goals, which is the subject of [[knowledge_base/research/social_cognition]].

The framework competes with established techniques, including reinforcement learning, Bayesian filtering and model predictive control, and the comparisons that have been reported are limited. The practical issues are the cost of inference over many policies, the need to specify or learn a generative model, and the choice of precision parameters. A reasonable claim is that Active Inference provides a coherent way to combine components that are otherwise assembled ad hoc; the evidence that it yields better performance is still thin.

## In this repository

Low-level control is covered in [[knowledge_base/research/robot_control]], and robotic implementations in [[knowledge_base/free_energy_principle/implementations/robotics]]. Conceptual background is in [[knowledge_base/cognitive/embodied_cognition]], [[knowledge_base/cognitive/enactive_cognition]], [[knowledge_base/cognitive/sensorimotor_coordination]] and [[knowledge_base/cognitive/epistemic_foraging]]. A study path is [[active_inference_robotics_learning_path]].

## References

- Lanillos, P., et al. (2021). Active inference in robotics and artificial agents: survey and challenges. arXiv:2112.01871.
- Pio-Lopez, L., Nizard, A., Friston, K., & Pezzulo, G. (2016). Active inference and robot control: a case study. *Journal of the Royal Society Interface*, 13, 20160616.

## Related Resources

- [[knowledge_base/research/robot_control]]
- [[knowledge_base/cognitive/active_inference]]
