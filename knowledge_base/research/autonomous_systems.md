---
title: Autonomous Systems
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
      - "[[knowledge_base/research/cognitive_robotics]]"
      - "[[knowledge_base/cognitive/multi_agent_active_inference]]"
      - "[[knowledge_base/cognitive/stigmergic_coordination]]"
      - "[[knowledge_base/cognitive/swarm_intelligence]]"
      - "[[knowledge_base/systems/fault_tolerance]]"
      - "[[knowledge_base/systems/resilient_systems]]"
      - "[[knowledge_base/systems/risk_assessment]]"
---

# Autonomous Systems

## Overview

An autonomous system performs tasks over extended periods with limited human supervision, deciding what to do from its own sensing. Examples include self-driving vehicles, uncrewed aerial vehicles (UAVs), underwater and planetary robots, and groups of robots coordinating without central control. Autonomy is graded, not binary: the SAE J3016 standard, for example, defines six levels of driving automation from none to full.

## Main questions

- **Perception under uncertainty.** Fusing sensors and maintaining a map and pose estimate.
- **Planning and decision-making** with partial observability, other agents and safety constraints.
- **Verification and assurance.** Showing that behaviour stays inside a safe envelope, including in situations not seen in testing.
- **Fault handling.** Detecting degraded sensors or actuators and degrading gracefully.
- **Coordination.** How several units divide work and avoid conflict, with or without communication.

Deployed vehicle stacks are built from conventional perception, prediction, planning and control modules, increasingly using learned components, and are validated through large-scale testing and engineering standards.

## Connection to Active Inference

The connection is an active research programme at the level of simulation and small robots. The framework's attraction for autonomy is that state estimation, planning and exploration come from one objective. Uncertainty is represented explicitly, so a vehicle can choose actions that reduce uncertainty about something safety-relevant, which is the epistemic term of expected free energy. Constraints such as keeping within a lane or avoiding obstacles can be expressed as strong prior preferences over outcomes. Navigation, active sensing and fault-aware control have all been studied in this form; see [[knowledge_base/research/robot_control]] for the control side.

For groups of robots, the relevant strand is multi-agent Active Inference, in which each unit holds a model that includes its neighbours and the shared environment. Stigmergic coordination, where agents communicate by modifying the environment as ants do with pheromone, is a related line and is documented in this repository. These models show that decentralised coordination can emerge from local inference in simulation. Large-scale real-world swarms using the approach have not been demonstrated, and comparison with established swarm-robotics controllers is limited.

Safety-critical deployment raises a point that the framework does not resolve by itself. A system with an explicit probabilistic model can report its uncertainty, which is useful, but the calibration of that uncertainty depends on the quality of the model, and formal assurance still relies on external verification and monitoring.

## In this repository

See [[knowledge_base/cognitive/multi_agent_active_inference]], [[knowledge_base/cognitive/stigmergic_coordination]] and [[knowledge_base/cognitive/swarm_intelligence]] for coordination, and [[knowledge_base/systems/fault_tolerance]], [[knowledge_base/systems/resilient_systems]] and [[knowledge_base/systems/risk_assessment]] for the engineering side. The Ant_Colony implementation under `code/Things` is a worked swarm example. A study path is [[active_inference_robotics_learning_path]].

## References

- SAE International. J3016: Taxonomy and Definitions for Terms Related to Driving Automation Systems for On-Road Motor Vehicles.

## Related Resources

- [[knowledge_base/research/robot_control]]
- [[knowledge_base/research/cognitive_robotics]]
