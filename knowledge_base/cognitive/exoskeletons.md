---
title: Exoskeletons
description: Wearable powered or passive frames that support or augment limb movement, their main designs and clinical uses, and their connection to motor control and brain-machine interfaces.
type: concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - rehabilitation
  - assistive_devices
  - motor_control
semantic_relations:
  - type: relates
    links:
      - "[[motor_control]]"
      - "[[orthoses]]"
      - "[[neuroprosthetics]]"
      - "[[brain_machine_interfaces]]"
      - "[[assistive_technology]]"
---

# Exoskeletons

## Overview
An exoskeleton is a wearable structure that is worn outside the body and that works in parallel with the user's limbs to support, assist or augment movement. Devices may be passive, using springs and mechanical linkages to store and return energy, or active, using electric, pneumatic or hydraulic actuators controlled by sensors. They are used for rehabilitation after stroke or spinal cord injury, for mobility assistance, and for reducing load and fatigue in occupational and military settings.

## Design and control
Dollar and Herr (2008) surveyed lower-limb exoskeletons and active orthoses and identified recurring engineering problems: power supply, actuator size and weight, alignment of the robot's joints with the user's anatomical joints, and the control strategy. Rigid frames transmit forces through the skeleton, while soft exosuits use textile structures and cable drives to apply forces more gently. Control approaches include:
- **Trajectory tracking**, in which the device drives the limb along a prescribed gait pattern.
- **Assist-as-needed control**, in which support is added only when the user's own effort falls short.
- **Intent-based control**, in which the device infers the wearer's goal from joint angles, interaction forces, muscle activity or, in research systems, brain signals (see [[brain_machine_interfaces]]).

## Clinical use
Robot-assisted gait training allows repeated, high-intensity practice for people with impaired walking. The motivation is activity-dependent plasticity (see [[neural_plasticity]] and [[motor_learning]]). Evidence on functional benefit differs between conditions and devices, and powered exoskeletons for walking after spinal cord injury have so far been used mostly in supervised settings.

## Relation to Active Inference
The link is an analogy and an engineering parallel. A device that infers the wearer's intended movement and adjusts its assistance resembles a second controller coupled to a biological one; its design problem has been described in terms of shared control and predictive models. Active Inference formulations of motor control, in which movement is produced by fulfilling proprioceptive predictions, give a way to describe the human side of this coupling, but no widely adopted exoskeleton controller is built on them.

## Related concepts
- [[orthoses]] — braces and supports that exoskeletons extend with actuation
- [[neuroprosthetics]] — devices that replace or augment neural function
- [[brain_machine_interfaces]] — a source of intent signals for control
- [[motor_control]] — the biological control problem being assisted
- [[assistive_technology]] — the broader category of supportive devices

## References
- Dollar, A. M., & Herr, H. (2008). Lower extremity exoskeletons and active orthoses: Challenges and state-of-the-art. IEEE Transactions on Robotics, 24, 144-158.
- Herr, H. (2009). Exoskeletons and orthoses: Classification, design challenges and future directions. Journal of NeuroEngineering and Rehabilitation, 6, 21.
