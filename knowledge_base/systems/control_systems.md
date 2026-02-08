---
title: Control Systems
type: concept
status: stub
created: 2026-02-06
tags:
  - systems-theory
  - control-theory
  - cybernetics
semantic_relations:
  - type: relates
    links:
      - [[dynamical_systems]]
      - [[adaptive_systems]]
      - [[circular_causality]]
---

# Control Systems

## Overview

Control systems theory studies how dynamical systems can be regulated to achieve desired behaviors. In the context of the Free Energy Principle, active inference provides a unified account of perception and action as control — where agents minimize the divergence between predicted and desired sensory states through both perceptual updating and motor commands.

## Key Concepts

### Feedback Control
- Closed-loop systems that adjust output based on error signals
- Negative feedback drives systems toward setpoints
- Prediction error in predictive coding as a control signal

### Optimal Control
- Minimization of cost functions over trajectories
- Relationship to active inference through the duality between control and inference
- [[knowledge_base/mathematics/path_integral_theory|Path integral]] approaches to stochastic optimal control

### Hierarchical Control
- Multi-level control architectures with cascading setpoints
- Perceptual control theory and hierarchical predictive coding
- Connection to [[knowledge_base/cognitive/hierarchical_inference|hierarchical inference]]

### Adaptive Control
- Controllers that modify their parameters online
- Model-reference adaptive control and its links to Bayesian inference
- Related to [[adaptive_systems]] and [[knowledge_base/cognitive/learning_mechanisms|learning]]

## Relationship to Active Inference

Active inference subsumes classical control theory by framing action selection as inference:
- **Control signal** ↔ **Action** that minimizes expected free energy
- **Reference signal** ↔ **Prior preference** (C matrix in POMDP formulation)
- **Error signal** ↔ **Prediction error** driving both perception and action
- **Plant model** ↔ **Generative model** of environment dynamics

## Related Topics

- [[dynamical_systems|Dynamical Systems]]
- [[adaptive_systems|Adaptive Systems]]
- [[circular_causality|Circular Causality]]
- [[knowledge_base/cognitive/active_inference|Active Inference]]
- [[knowledge_base/mathematics/optimization_theory|Optimization Theory]]

---

> [!note] Open Source and Licensing
> Repository: [ActiveInferenceInstitute/cognitive](https://github.com/ActiveInferenceInstitute/cognitive)
> - Documentation and knowledge base content: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
> - Code and examples: MIT License (see `LICENSE`)
