---
title: Collision Avoidance
description: How moving animals, groups and robots steer to avoid hitting neighbours or obstacles, covering separation rules, looming detection and social-force models, and how avoidance can be framed through preferences.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - collective_behavior
  - locomotion
  - navigation
semantic_relations:
  - type: relates
    links:
      - "[[swarm_intelligence]]"
      - "[[bird_flocks]]"
      - "[[v_formation]]"
      - "[[motion_perception]]"
      - "[[robot_control]]"
      - "[[collective_behavior]]"
---

# Collision Avoidance

## Overview
Collision avoidance is the capacity of a moving agent to change its path so as not to hit other agents or obstacles. It is a basic requirement for animals moving in groups, for people walking in crowds, and for mobile robots. In collective motion it is usually the short-range component of a balance of rules, with attraction and alignment acting at longer ranges.

## Models
In Reynolds' influential computer model of flocking (1987), each simulated bird follows three local steering rules: separation from nearby neighbours, alignment with their heading, and cohesion toward their centre. Separation is the collision avoidance rule. Similar zones of repulsion, orientation and attraction appear in many models of fish schools and bird flocks. Social-force models of pedestrian dynamics, introduced by Helbing and Molnár, treat each person as responding to repulsive forces from others and obstacles and an attractive force toward a goal. In robotics, reactive potential-field and velocity-based methods compute safe velocities from the relative motion of neighbours.

## Perception and behaviour
Detecting an imminent collision depends on visual cues such as the rate of expansion of an approaching object's image, called looming. Neurons tuned to looming, studied in locusts, respond as an object approaches and are linked to escape and avoidance. Studies of human walking show that people steer around obstacles and moving targets in ways that can be described by simple dynamical models in which heading responds to the angle and distance of the obstacle, not to a global plan (Fajen and Warren, 2003).

## Relation to Active Inference
In Active Inference, avoidance can be expressed by giving the agent strongly dispreferred outcomes for collisions in its prior preferences, so that policies predicting contact have high expected free energy. This is how some robotic and navigation models are formulated. It is a modelling choice that reproduces avoidance, not a demonstration that animals or crowds actually compute in this way, and the established models mentioned above are largely kinematic or force-based.

## Related concepts
- [[bird_flocks]] — a main example of collective motion with avoidance
- [[v_formation]] — formation flight, where spacing is maintained
- [[swarm_intelligence]] — local rules producing group-level order
- [[motion_perception]] — visual detection of approaching objects
- [[robot_control]] — engineering approaches to safe navigation

## References
- Reynolds, C. W. (1987). Flocks, herds and schools: a distributed behavioral model. Computer Graphics (SIGGRAPH '87 Proceedings), 21(4), 25-34.
- Helbing, D., & Molnár, P. (1995). Social force model for pedestrian dynamics. Physical Review E, 51, 4282-4286.
- Fajen, B. R., & Warren, W. H. (2003). Behavioral dynamics of steering, obstacle avoidance, and route selection. Journal of Experimental Psychology: Human Perception and Performance, 29, 343-362.
