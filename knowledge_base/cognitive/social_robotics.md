---
title: Social Robotics
description: The design and study of robots that interact with people through social cues such as gaze, speech and gesture, including major research areas, challenges, and links to Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - robotics
  - human_robot_interaction
  - social_cognition
semantic_relations:
  - type: relates
    links:
      - "[[human_robot_interaction]]"
      - "[[robotics]]"
      - "[[cognitive_robotics]]"
      - "[[developmental_robotics]]"
      - "[[social_cognition]]"
---

# Social Robotics

## Overview
Social robotics is the field concerned with robots designed to interact and communicate with people, and sometimes with other robots, in ways that follow human social conventions. Such robots use signals such as gaze, facial expression, gesture, speech and turn-taking to be understood by people without specialised training. Applications that have been studied include education, therapy support, elder care, reception and companionship. Fong, Nourbakhsh and Dautenhahn (2003) surveyed socially interactive robots and distinguished several families, including socially evocative, social interface, sociable and socially intelligent designs.

## Research areas
- **Perception of people**: detecting faces, speech, gaze, emotion and activity in real time.
- **Expressive behaviour**: robot faces, gestures, voice and gaze that convey intention or state. Breazeal (2003) reviewed approaches to building sociable robots.
- **Dialogue and turn-taking**: managing timing and shared attention (see [[joint_attention]]).
- **Social learning**: learning from demonstration and imitation, drawing on [[developmental_robotics]].
- **Human factors**: trust, acceptance and expectations, including the tendency to attribute mental states to machines. Reactions to near-human appearance, sometimes called the uncanny valley, have been reported but results are mixed.

Open problems include robust behaviour in unstructured settings, long-term engagement beyond novelty, privacy, and ethical questions about deception, dependence and appropriate use with vulnerable people. Evaluations are often short, small and conducted in laboratories, which limits what can be concluded about real-world benefit.

## Relation to Active Inference
Active Inference has been applied to robot control and perception, where a robot minimises prediction error or free energy to reach goals, and some work extends this to social settings in which a robot maintains a generative model of a human partner, infers the partner's goals and plans accordingly. This is an active research programme, with proof-of-concept demonstrations in simulation and on hardware, but it is not a standard approach in the field compared with learning-based and rule-based methods. See [[robot_control]] and [[human_robot_interaction]].

## Related concepts
- [[human_robot_interaction]] — the broader study of people and robots
- [[robotics]] — general field of robot design
- [[cognitive_robotics]] — robots with cognitive architectures
- [[developmental_robotics]] — robot learning modelled on development
- [[swarm_robotics]] — multi-robot coordination, with different social structure
- [[social_cognition]] — human processes robots must engage with

## References
- Fong, T., Nourbakhsh, I., & Dautenhahn, K. (2003). A survey of socially interactive robots. Robotics and Autonomous Systems, 42, 143-166.
- Breazeal, C. (2003). Toward sociable robots. Robotics and Autonomous Systems, 42, 167-175.
- Goodrich, M. A., & Schultz, A. C. (2007). Human-robot interaction: A survey. Foundations and Trends in Human-Computer Interaction, 1, 203-275.
