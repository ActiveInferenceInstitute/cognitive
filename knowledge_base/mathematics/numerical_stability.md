---
title: Numerical Stability in Inference and Control
description: "Short reference on numerical stability techniques for inference and control, including log-space computation, log-sum-exp, epsilon floors, gradient clipping and bounded precision."
type: mathematical_concept
status: stable
created: 2025-08-08
tags:
- numerics
- stability
- optimization
semantic_relations:
  - type: supports
    links:
      - "[[variational_inference]]"
      - "[[policy_selection]]"
      - "[[action_distribution]]"
  - type: relates
    links:
      - "[[softmax_function]]"
      - "[[precision_parameter]]"
---


## Numerical Stability

### Common techniques

- Log-space computations for products/sums of probabilities

- Log-sum-exp and max-subtraction for [[softmax_function]]

- Epsilon floors/ceils to avoid log(0) and divide-by-zero

- Gradient clipping and parameter damping

- Positive-definiteness enforcement (e.g., jitter on covariances)

### In this repository

- Stabilizing EFE/VFE computations and policy softmax

- Bounding [[precision_parameter]] (γ) and normalizing value scales

### See also

- [[softmax_function]] · [[precision_parameter]] · [[variational_inference]]

