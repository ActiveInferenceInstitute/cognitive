---
title: Kinetics
description: The relation between forces or torques and the resulting motion of bodies, from Newton's laws to rigid-body manipulator dynamics and muscle force models, as used in motor control.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - mechanics
  - dynamics
  - motor_control
semantic_relations:
  - type: relates
    links:
      - "[[kinematics]]"
      - "[[motor_control]]"
      - "[[biomechanics]]"
      - "[[lagrangian_mechanics]]"
      - "[[optimal_control]]"
wikidata: Q1108792
---

# Kinetics

## Overview
In mechanics, kinetics is the study of how forces and torques produce or change motion, in contrast with [[kinematics]], which describes motion alone. It rests on Newton's second law, $\mathbf{F}=m\mathbf{a}$ for a point mass and $\tau=I\alpha$ for rotation about a fixed axis, where $I$ is the moment of inertia. The word has a different meaning in chemistry, where chemical kinetics concerns reaction rates; the sense here is mechanical.

## Dynamics of articulated bodies
For a multi-joint limb or robot arm with joint angles $q$, the equations of motion take the form
$$M(q)\ddot q+C(q,\dot q)\dot q+g(q)=\tau,$$
where $M$ is the symmetric positive-definite inertia matrix, $C\dot q$ collects Coriolis and centripetal terms, $g$ is gravity and $\tau$ is the vector of joint torques. They can be derived from the Newton-Euler method or from the Euler-Lagrange equations of the kinetic and potential energy. Two problems are distinguished: forward dynamics computes the motion produced by given torques, and inverse dynamics computes the torques needed to realise a given motion. The coupling between joints means that moving one segment produces torques at the others.

## Muscle and force generation
In biological limbs torques are produced by muscles, whose force depends on activation, length and shortening velocity. Hill (1938) characterised the force-velocity relation of shortening muscle, and Hill-type models combine this with a length-tension curve and an elastic element. Redundancy appears again: many muscle activation patterns give the same net joint torque, so optimisation criteria or synergies are used to select among them.

## Relation to Active Inference
Classical optimal control computes inverse dynamics, then forces, from a desired trajectory. Friston (2011) argued that Active Inference removes the need for explicit inverse models and cost functions: descending signals specify proprioceptive predictions, and reflex arcs resolve proprioceptive prediction errors by producing the forces that fulfil them. The kinetic details are absorbed into the generative process, which the agent does not need to model exactly. This is a theoretical proposal with working simulations, and the extent to which it matches neural implementation remains under study.

## Related concepts
- [[kinematics]] — motion described without forces
- [[motor_control]] — the control of movement by the nervous system
- [[biomechanics]] — application of mechanics to organisms
- [[optimal_control]] — computing torques by minimising a cost
- [[lagrangian_mechanics]] — an energy-based derivation of the equations of motion
- [[muscle_synergies]] — low-dimensional organisation of muscle activation

## References
- Goldstein, H., Poole, C., & Safko, J. (2002). Classical Mechanics (3rd ed.). Addison-Wesley.
- Hill, A. V. (1938). The heat of shortening and the dynamic constants of muscle. Proceedings of the Royal Society of London B, 126, 136-195.
- Friston, K. (2011). What is optimal about motor control? Neuron, 72, 488-498.
