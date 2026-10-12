---
title: Collective Transport
description: Cooperative carrying of loads too large for one individual, as seen in ants and implemented in groups of robots, and the coordination mechanisms and control problems it involves.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - collective_behavior
  - social_insects
  - robotics
semantic_relations:
  - type: relates
    links:
      - "[[swarm_robotics]]"
      - "[[swarm_intelligence]]"
      - "[[collective_behavior_ants]]"
      - "[[task_allocation]]"
      - "[[stigmergic_coordination]]"
      - "[[joint_action]]"
---

# Collective Transport

## Overview
Collective transport occurs when several individuals jointly move an object that none of them could move alone. It is well documented in ants, where groups carry large prey or food items back to the nest, and it is a standard task in multi-robot research because it needs coordination of force and direction without necessarily needing central control.

## In ants
Species in several genera retrieve large items cooperatively. A group typically has to solve problems of recruitment, grasping, steering and handling obstacles. Individuals can pull or lift at different positions around the load. Direction is not usually decided by a leader with a plan: it emerges from the sum of individual efforts and from mechanical coupling through the load, which carries information about the others' pulls. In some species, informed individuals who know the route can bias the group's direction, and carriers may realign when the load becomes stuck. Cooperative transport by ants has been analysed with physical models that treat the load as a shared mechanical medium (see [[collective_behavior_ants]]).

## In robots
Robot groups have been used for box pushing, pulling and carrying, mostly by simple local rules and little or no communication. Kube and Bonabeau showed that a group of robots using simple behaviour could push a box toward a goal and recover from stagnation, taking ants as an inspiration. Designs differ in whether robots grasp, push or cage the object, and in whether they use explicit communication, force sensing or coordination through the object.

## Relation to Active Inference
Collective transport is a clear example of joint action: agents infer each other's intentions from their effect on a shared object. Multi-agent Active Inference models of joint action and shared goals exist in theory, and the load can be treated as a shared sensory channel. However, the classic work on ants and robots uses behavioural and control-theoretic models, so the link is a modelling option rather than an established explanation.

## Related concepts
- [[swarm_robotics]] — multi-robot systems designed on swarm principles
- [[swarm_intelligence]] — the broader field of decentralised coordination
- [[task_allocation]] — division of work among colony members
- [[stigmergic_coordination]] — coordination through the environment
- [[joint_action]] — coordinated action between agents in cognitive science

## References
- Kube, C. R., & Bonabeau, E. (2000). Cooperative transport by ants and robots. Robotics and Autonomous Systems, 30, 85-101.
- Hölldobler, B., & Wilson, E. O. (1990). The Ants. Harvard University Press.
- Şahin, E. (2005). Swarm robotics: from sources of inspiration to domains of application. In Swarm Robotics (Lecture Notes in Computer Science, vol. 3342, pp. 10-20). Springer.
