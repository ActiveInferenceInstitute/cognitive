---
title: Particle Filtering
description: Sequential Monte Carlo methods that approximate filtering distributions of nonlinear, non-Gaussian state-space models with weighted samples, covering the bootstrap filter, resampling, and links to sampling accounts of the brain.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - state_estimation
  - monte_carlo
  - sequential_inference
semantic_relations:
  - type: relates
    links:
      - "[[state_estimation]]"
      - "[[monte_carlo_methods]]"
      - "[[importance_sampling]]"
      - "[[bayesian_brain]]"
      - "[[variational_inference]]"
---

# Particle Filtering

## Overview
Particle filtering, also called sequential Monte Carlo, estimates the hidden state of a dynamical system from a stream of noisy observations when the model is nonlinear or non-Gaussian, so that the Kalman filter's closed-form solution does not apply. The filtering distribution $p(x_t\mid y_{1:t})$ is represented by a set of $N$ samples (particles) $x_t^{(i)}$ with weights $w_t^{(i)}$, and is updated recursively as each observation arrives.

## Algorithm
The state-space model specifies a transition density $p(x_t\mid x_{t-1})$ and an observation density $p(y_t\mid x_t)$. The bootstrap filter of Gordon, Salmond and Smith (1993) proceeds in three steps at each time point:
1. **Propagate**: draw each particle from the transition density, $x_t^{(i)}\sim p(x_t\mid x_{t-1}^{(i)})$.
2. **Weight**: set $w_t^{(i)}\propto p(y_t\mid x_t^{(i)})$ and normalise, so that particles consistent with the observation count for more. This is [[importance_sampling]] with the transition density as proposal.
3. **Resample**: draw $N$ particles with probability proportional to the weights, which removes low-weight particles and duplicates high-weight ones.

Without resampling the weights degenerate, with a single particle carrying nearly all of the mass. Resampling reduces that problem but adds Monte Carlo variance, and designs differ in when and how they resample. Better proposal distributions that use the current observation reduce variance. The approximation converges to the true filtering distribution as $N\to\infty$ under mild conditions, but the number of particles needed can grow rapidly with the state dimension, so the method works best in low to moderate dimensions.

## Relation to Active Inference and the Bayesian brain
Particle methods are an alternative inference scheme, not the standard one used in Active Inference. Most Active Inference models use variational approximations, with Gaussian beliefs under the Laplace assumption in continuous time and mean-field or marginal message passing in discrete time. Sampling-based accounts of neural computation, in which variability in neural activity represents samples from a posterior, form a separate hypothesis within the Bayesian brain literature (see [[bayesian_brain]]), and particle filters have been proposed as one algorithmic model of how such sampling could be organised. The evidence for these proposals is mixed, and the question of whether brains sample or use parametric codes remains open. Variational and sampling approaches trade accuracy against cost in different ways (see [[variational_inference]]).

## Related concepts
- [[state_estimation]] — the task particle filters perform
- [[monte_carlo_methods]] — the wider family of sampling-based inference
- [[importance_sampling]] — the weighting step of the filter
- [[bayesian_brain]] — sampling hypotheses about neural computation
- [[variational_inference]] — the deterministic alternative used in Active Inference

## References
- Gordon, N. J., Salmond, D. J., & Smith, A. F. M. (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. IEE Proceedings F, 140, 107-113.
- Doucet, A., de Freitas, N., & Gordon, N. (Eds.). (2001). Sequential Monte Carlo Methods in Practice. Springer.
