---
title: Online Learning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - learning
  - machine_learning
  - sequential
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[belief_updating]]"
      - "[[bayesian_inference]]"
---

# Online Learning

## Overview
Online learning refers to updating a model one observation (or small mini-batch) at a time as data arrive, instead of fitting to a fixed dataset in a single batch pass. The learner makes a prediction, receives feedback, and adjusts before the next input. It is the natural regime for agents that interact continuously with an environment and cannot revisit past data freely.

## Mechanism
In optimization terms, online learning includes stochastic gradient descent on a stream, $\theta_{t+1} = \theta_t - \eta_t \nabla \ell(\theta_t; x_t)$, and online convex optimization, which studies **regret**, the cumulative loss relative to the best fixed comparator in hindsight. Bayesian online learning replaces the point estimate with a posterior that is updated recursively:

$$p(\theta \mid x_{1:t}) \propto p(x_t \mid \theta)\, p(\theta \mid x_{1:t-1}).$$

Last step's posterior becomes this step's prior. Key practical issues are:

- **Learning-rate schedule**: a fixed rate tracks change but never settles; a decaying rate converges but adapts slowly.
- **Non-stationarity**: when the data-generating process drifts, forgetting or change-point detection is needed.
- **Catastrophic forgetting**: sequential training of neural networks on shifting data can overwrite earlier knowledge, the stability-plasticity problem (see [[stability_plasticity]]).
- **Single pass**: no shuffling or repeated epochs, so the learner depends on update rules that are stable under correlated inputs.

## Role in Active Inference
Active inference is online by construction. An agent updates beliefs about hidden states at each time step by minimizing free energy, and, on a slower timescale, updates generative-model parameters. For discrete models the parameter update is a running count of Dirichlet concentrations, which is exactly the recursive Bayesian update above, so the learning rate is not a free hyperparameter: it falls as evidence accumulates because each added count shifts a larger total less. This is a useful property but also a limitation, since the model cannot adapt to a changing world unless a forgetting term or a mechanism that resets or expands the model is added (see [[metaplasticity]], [[structure_learning]]). The online character is a statement about the update scheme, not about the free-energy objective: batch variational inference optimizes the same bound. The brain's learning operates online and under resource limits, which is why local, incremental rules are favored in biologically motivated implementations.

## Related concepts
- [[belief_updating]] — the per-step inference that online learning builds on
- [[bayesian_inference]] — recursive posterior updating
- [[stability_plasticity]] — the forgetting trade-off in sequential learning
- [[meta_learning]] — learning how to adapt learning rates and priors
- [[reinforcement_learning]] — a canonical online setting
- [[transfer_learning]] — reuse across tasks, another route to faster adaptation
