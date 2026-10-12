---
title: Human-Robot Interaction
description: The study and design of communication and collaboration between people and robots, covering interaction modes, social attribution to machines, the contested uncanny valley, and framings from joint action and Active Inference.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - robotics
  - social_interaction
  - human_factors
semantic_relations:
  - type: relates
    links:
      - "[[social_robotics]]"
      - "[[social_cognition]]"
      - "[[human_computer_interaction]]"
      - "[[joint_action]]"
      - "[[theory_of_mind]]"
---

# Human-Robot Interaction

## Overview
Human-robot interaction (HRI) is the field concerned with understanding, designing and evaluating robotic systems that are used by, or work alongside, people. It combines robotics, human-computer interaction, psychology and design. Interaction ranges from remote supervision of a machine, through shared physical workspaces such as assembly and surgery, to social settings in which a robot converses with or accompanies a person (Goodrich & Schultz, 2007).

## Channels and design problems
People coordinate with one another through speech, gaze, gesture, posture and touch, and HRI research asks which of these a robot should produce and interpret. A recurring design problem is legibility: a robot's movements should let an observer infer its goal early, rather than merely being efficient. A second problem is trust calibration, since over-trust leads to misuse of a robot and under-trust leads to disuse. A third is timing, because delays in turn-taking are noticed quickly in conversation. Research on social robots, such as Breazeal's work on expressive robot heads, treats appearance and expressive behaviour as ways of making an agent's state readable to a human partner (Breazeal, 2003).

## Attribution and the uncanny valley
People readily ascribe intentions and feelings to machines that move or respond contingently. This tendency is robust, but its consequences are less settled. Mori (1970) proposed that affinity for a robot rises with human likeness and then drops sharply for near-human but imperfect appearance. The hypothesis has been influential, yet empirical support is mixed: some studies find a dip in ratings, others do not, and the outcome depends on the stimuli, the dimension rated and the measure used. It is best regarded as a hypothesis about a family of effects rather than an established law.

## Relation to Active Inference
HRI can be described as two agents, each equipped with a generative model of the other, trying to reduce uncertainty about shared goals. Legible motion then corresponds to behaviour that makes a partner's inference about intent easier, and robot controllers have been built on active inference principles. The link to HRI itself is mainly an interpretive framework; controlled human studies that test active inference predictions against alternatives in HRI remain limited.

## Related concepts
- [[social_robotics]] — robots designed for social roles
- [[joint_action]] — the human model for coordinated action with a robot partner
- [[theory_of_mind]] — inferring a partner's goals and beliefs
- [[embodied_communication]] — non-verbal channels used in interaction
- [[robot_control]] — the engineering side of acting in shared spaces

## References
- Goodrich, M. A., & Schultz, A. C. (2007). Human-robot interaction: A survey. Foundations and Trends in Human-Computer Interaction, 1(3), 203-275.
- Breazeal, C. (2003). Toward sociable robots. Robotics and Autonomous Systems, 42, 167-175.
- Mori, M. (2012). The uncanny valley (K. F. MacDorman & N. Kageki, Trans.). IEEE Robotics & Automation Magazine, 19(2), 98-100. (Original work published 1970)
