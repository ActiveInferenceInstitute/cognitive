---
title: Planning as Inference
type: concept
status: stable
created: 2025-03-03
updated: 2025-12-18
tags:
  - planning
  - active_inference
  - decision_making
  - probabilistic_inference
semantic_relations:
  - type: implements
    links:
      - [[active_inference]]
      - [[../mathematics/expected_free_energy]]
  - type: relates
    links:
      - [[decision_making]]
      - [[hierarchical_inference]]
      - [[../mathematics/active_inference_pomdp]]
  - type: foundation
    links:
      - [[../mathematics/bayesian_inference]]
      - [[../mathematics/optimal_control]]

      - [[active_inference]]

      - [[../mathematics/expected_free_energy]]

      - [[../mathematics/policy_selection]]

      - [[predictive_coding]]

      - [[../mathematics/variational_free_energy]]

---

## Planning as Inference

## Overview

Planning as inference casts action selection as probabilistic inference under a generative model. Desired outcomes are encoded as priors (preferences) and plans (policies) are inferred by minimizing expected free energy.

## Core Formulation

```math

P(\pi) \propto \exp\big(-\gamma\,G(\pi)\big),\quad G(\pi)=\sum_{\tau} G(\pi,\tau)

```

- [[../mathematics/expected_free_energy]]: balances epistemic and pragmatic value

- [[../mathematics/policy_selection]]: softmax over negative EFE

- [[../mathematics/variational_free_energy]]: perceptual inference objective

## Connections

- [[active_inference]]: unified scheme for perception and action

- [[../cognitive/predictive_coding]]: message passing implementation

- [[../mathematics/active_inference_pomdp]]: discrete POMDP instantiation

## See Also

- [[../mathematics/epistemic_value]]

- [[../mathematics/pragmatic_value]]

