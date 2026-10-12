---
title: Learning Rate
description: The step-size parameter that scales parameter updates in gradient-based and incremental learning, its convergence conditions, adaptive schedules, and its counterparts in Bayesian filtering and Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - optimization
  - learning
  - gradient_descent
semantic_relations:
  - type: relates
    links:
      - "[[gradient_descent]]"
      - "[[adaptive_step_size]]"
      - "[[temperature_parameter]]"
      - "[[parameter_learning]]"
      - "[[precision_parameter]]"
wikidata: Q65121812
---

# Learning Rate

## Overview
A learning rate is a positive scalar $\eta$ that controls how far parameters move in response to an error signal. In gradient descent on a loss $L$, the update is
$$\theta_{k+1}=\theta_k-\eta\,\nabla L(\theta_k).$$
The same idea appears in incremental estimators, where an estimate moves toward each new observation by a fraction $\alpha$ of the discrepancy, and in temporal-difference methods in reinforcement learning.

## Choosing the step size
A value that is too small makes progress slow, and one that is too large causes oscillation or divergence. For a quadratic loss whose Hessian has largest eigenvalue $\lambda_{\max}$, gradient descent converges if and only if $0<\eta<2/\lambda_{\max}$, and $\eta=1/\lambda_{\max}$ is a common safe choice. For stochastic gradients, the Robbins-Monro conditions $\sum_k\eta_k=\infty$ and $\sum_k\eta_k^2<\infty$ guarantee convergence under standard assumptions: the steps are large enough to reach the optimum but shrink fast enough to average out noise. In practice a decay schedule, warm-up, or an adaptive method such as Adam, which rescales the step per parameter, is often used. Constant rates remain common when a learner must track a changing environment, because they weight recent data more heavily.

## Relation to Bayesian filtering
In a Kalman filter the gain plays the role of a learning rate, and it is set by the ratio of prior uncertainty to total uncertainty. A noisy observation therefore produces a small update and a reliable one a large update. This shows that an optimal learning rate depends on estimated uncertainty, not on a fixed constant.

## Relation to Active Inference
The connection is direct in continuous-time schemes, where belief updates are gradient flows on free energy weighted by precision. Precision then acts as an adaptive, context-dependent gain on prediction errors, which parallels the Kalman gain. In discrete-state models, learning of likelihood and transition parameters is usually written as accumulation of Dirichlet concentration counts, and a learning rate scales how strongly each observation contributes. Whether the brain implements an explicit adaptive rate is an empirical question under investigation in computational psychiatry and neuroscience.

## Related concepts
- [[gradient_descent]] — the optimisation procedure in which the rate scales each step
- [[adaptive_step_size]] — methods that adjust the step during optimisation
- [[temperature_parameter]] — another scalar that governs how sharply updates or choices respond to values
- [[precision_parameter]] — uncertainty-based gain on errors
- [[parameter_learning]] — learning model parameters from data

## References
- Robbins, H., & Monro, S. (1951). A stochastic approximation method. Annals of Mathematical Statistics, 22, 400-407.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
