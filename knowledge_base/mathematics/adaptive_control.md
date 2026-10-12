---
title: Adaptive Control
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - adaptive_control
  - parameter_estimation
  - lyapunov
semantic_relations:
  - type: foundation
    links:
      - "[[feedback_control]]"
      - "[[control_theory]]"
  - type: relates
    links:
      - "[[stability_analysis]]"
      - "[[active_inference_for_control]]"
      - "[[allostatic_control]]"
wikidata: Q235781
---

# Adaptive Control

## Overview
Adaptive control designs controllers that adjust their own parameters online to cope with plant parameters that are unknown or drifting. The controller therefore has two coupled loops: a fast loop that regulates the plant and a slower loop that estimates or tunes parameters.

## Definition / Formalism
Two main architectures are model reference adaptive control (MRAC), which drives the plant output toward the output of a reference model, and self-tuning regulators, which combine a recursive parameter estimator with a control law designed as if the estimate were exact (the certainty-equivalence principle).

Scalar example. Plant $\dot x=ax+u$ with $a$ unknown, reference dynamics $\dot x=-a_m x$ with $a_m>0$. Use $u=-(\hat a+a_m)x$ and the update $\dot{\hat a}=\gamma x^2$ with $\gamma>0$. With $\tilde a=\hat a-a$ the closed loop is $\dot x=-a_mx-\tilde a x$. Take $V=\tfrac12x^2+\tfrac1{2\gamma}\tilde a^2$; then

$$\dot V=-a_mx^2-\tilde ax^2+\tilde a x^2=-a_mx^2\le0,$$

so $x$ and $\hat a$ stay bounded and, by Barbalat's lemma, $x\to0$. The parameter estimate need not converge to $a$ unless the signals are persistently exciting; a simulation with $a=1$, $a_m=2$, $\gamma=5$ drives $x$ to zero while $\hat a$ settles near 1.45, not 1. Practical schemes add robustness modifications (projection, dead zones, sigma-modification) against noise and unmodelled dynamics, which can otherwise cause parameter drift and instability.

## Role in Active Inference
Active inference has a similar two-timescale structure: fast belief updating about states and slower learning of parameters, both as gradient descent on the same free energy, with action closing the loop. This resemblance is structural, and I would hold it as an analogy rather than an equivalence. Differences: certainty-equivalent adaptive control uses point estimates of parameters, while active inference carries uncertainty about them through precisions and, in the discrete case, Dirichlet concentration parameters, so exploration of uncertain parameters can be driven by expected information gain. Formal treatments that recast control problems in active inference terms are collected in the control-oriented page of this vault.

## Related concepts
- [[feedback_control]] — the fixed-parameter loop that adaptive control extends
- [[stability_analysis]] — Lyapunov methods used to certify adaptation laws
- [[active_inference_for_control]] — control applications of active inference
- [[allostatic_control]] — predictive regulation with changing set points
- [[learning_mechanisms]] — parameter learning in cognitive models

## References
- Astrom & Wittenmark (1989), Adaptive Control.
- Ioannou & Sun (1996), Robust Adaptive Control.
