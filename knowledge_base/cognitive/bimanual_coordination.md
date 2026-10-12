---
title: Bimanual Coordination
description: Coordinated use of both hands, the stability of in-phase and anti-phase patterns, the Haken-Kelso-Bunz model of phase transitions, and neural mechanisms of intermanual coupling.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - motor_control
  - coordination_dynamics
  - dynamical_systems
semantic_relations:
  - type: relates
    links:
      - "[[motor_control]]"
      - "[[dynamical_systems]]"
      - "[[synchronization]]"
      - "[[supplementary_motor]]"
      - "[[sensorimotor_coordination]]"
---

# Bimanual Coordination

## Overview
Bimanual coordination is the control of movements of the two hands so that they work together, as in tying shoelaces, playing a piano or clapping. The central observation is that the hands do not move independently. Movements of one hand influence the other, and some spatiotemporal relations between the hands are performed readily while others are difficult without practice.

## Phase relations and stability
In rhythmic tasks, such as flexing and extending the index fingers, two patterns are stable without training. In-phase movement, with homologous muscles active together, and anti-phase movement, with alternating activation, can both be performed at low speeds. Intermediate phase relations are less stable and variable. When the movement frequency is increased from an anti-phase start, participants spontaneously switch to in-phase movement, and the reverse switch does not occur (Kelso, 1984).

## The Haken-Kelso-Bunz model
Haken, Kelso and Bunz (1985) described this behaviour with a coupling between two nonlinear oscillators. The relative phase $\phi$ serves as a collective variable, and its dynamics can be written as the gradient of a potential,
$$\dot\phi = -a\sin\phi - 2b\sin 2\phi,\qquad V(\phi)=-a\cos\phi-b\cos 2\phi,$$
where the ratio $b/a$ decreases as movement frequency increases. For large $b/a$ the potential has minima at $\phi=0$ and $\phi=\pi$. As the ratio falls below $1/4$, the anti-phase minimum disappears and only in-phase remains. The model explains the switch as a phase transition, with critical slowing and enhanced fluctuations before it, as predicted and observed (see [[phase_transitions]] and [[order_parameters]]).

## Neural basis
Intermanual coupling involves both peripheral and central factors. Cortical areas including the [[supplementary_motor]] and [[premotor_cortex]] regions, primary motor cortex, the [[cerebellum]] and the connections between hemispheres contribute, and interhemispheric inhibition and callosal pathways are implicated (Swinnen, 2002). Perceptual coupling matters as well, since the stability of patterns depends on how movements are perceived and not only on muscle commands. Ageing, stroke and disorders affecting the basal ganglia can impair bimanual coordination.

## Relation to Active Inference
The link is limited. The Haken-Kelso-Bunz description is a dynamical systems model, and its collective variable and potential can be formally related to the attractor and free-energy-landscape language used in Active Inference (see [[free_energy_landscape]]). Active Inference accounts of movement based on proprioceptive predictions and of joint action between agents share the idea of coupled systems settling into stable patterns (see [[joint_action]]). The coordination dynamics literature has not generally been formulated in terms of the free energy principle.

## Related concepts
- [[motor_control]] — general principles of movement control
- [[eye_hand_coordination]] — coordination between gaze and hand
- [[synchronization]] — coupling of oscillators
- [[dynamical_systems]] — mathematical framework for coordination dynamics
- [[motor_cortex]] — cortical control of the hands

## References
- Kelso, J. A. S. (1984). Phase transitions and critical behavior in human bimanual coordination. American Journal of Physiology: Regulatory, Integrative and Comparative Physiology, 246, R1000-R1004.
- Haken, H., Kelso, J. A. S., & Bunz, H. (1985). A theoretical model of phase transitions in human hand movements. Biological Cybernetics, 51, 347-356.
- Swinnen, S. P. (2002). Intermanual coordination: From behavioural principles to neural-network interactions. Nature Reviews Neuroscience, 3, 348-359.
