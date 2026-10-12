---
title: Kinematics
description: The description of motion through position, velocity and acceleration without reference to forces, including forward and inverse kinematics of articulated limbs and its use in motor control.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - mechanics
  - motor_control
  - robotics
semantic_relations:
  - type: relates
    links:
      - "[[kinetics]]"
      - "[[motor_control]]"
      - "[[biomechanics]]"
      - "[[robotics]]"
      - "[[trajectory_optimization]]"
---

# Kinematics

## Overview
Kinematics describes how bodies move, using position, velocity and acceleration as functions of time, and sets aside the forces that cause the motion. For constant acceleration $a$ along a line, $x(t)=x_0+v_0t+\tfrac12at^2$ and $v(t)=v_0+at$. For articulated bodies such as limbs or robot arms the main objects are the joint variables and the position of the end point.

## Forward and inverse kinematics
Let $q\in\mathbb{R}^n$ collect the joint angles of a manipulator and $x\in\mathbb{R}^m$ describe the pose of the hand. The forward kinematic map is $x=f(q)$. For a planar arm with two links of lengths $l_1,l_2$ and joint angles $q_1,q_2$,
$$x=l_1\cos q_1+l_2\cos(q_1+q_2),\qquad y=l_1\sin q_1+l_2\sin(q_1+q_2).$$
Velocities are related by the Jacobian $J(q)=\partial f/\partial q$ through $\dot x=J(q)\dot q$. Inverse kinematics asks for $q$ given $x$. It can have no solution, one, or many, and is redundant when $n>m$, as for a human arm reaching to a point in space. Redundancy is often resolved with the pseudo-inverse, $\dot q=J^{+}\dot x$, which picks the minimum-norm joint velocity, or by adding a secondary objective. Near singular configurations, where $J$ loses rank, joint velocities may become very large.

## Kinematic regularities in movement
Human reaching movements show approximately straight hand paths and bell-shaped speed profiles. Flash and Hogan (1985) showed that these features are predicted by minimising the integral of squared jerk, the third derivative of position. Such regularities suggest that planning is done in task or hand space as well as in joint space, although the interpretation is debated.

## Relation to Active Inference
In Active Inference accounts of motor control, desired trajectories are expressed as predictions about proprioceptive and exteroceptive input, and the descending signals are predictions rather than motor commands. Kinematic quantities such as joint angles and hand position are then the variables in which those predictions are formulated, with generative models mapping hidden causes to joint configurations. This is an active modelling approach whose empirical support is still being assessed.

## Related concepts
- [[kinetics]] — the companion description of the forces and torques producing motion
- [[motor_control]] — how the nervous system generates movement
- [[biomechanics]] — mechanics applied to biological bodies
- [[forward_models]] — predicting the sensory consequences of movement
- [[robotics]] — where kinematic analysis of manipulators was developed
- [[trajectory_optimization]] — choosing paths by optimising a cost

## References
- Craig, J. J. (2005). Introduction to Robotics: Mechanics and Control (3rd ed.). Pearson Prentice Hall.
- Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G. (2009). Robotics: Modelling, Planning and Control. Springer.
- Flash, T., & Hogan, N. (1985). The coordination of arm movements: an experimentally confirmed mathematical model. Journal of Neuroscience, 5, 1688-1703.
