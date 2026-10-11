---
title: State Space Models
description: Models in which a hidden state evolves over time and generates noisy observations, covering the linear Gaussian case with the Kalman filter, hidden Markov models, and their place in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - time_series
  - filtering
  - generative_models
semantic_relations:
  - type: relates
    links:
      - "[[hidden_states]]"
      - "[[state_estimation]]"
      - "[[markov_models]]"
      - "[[generative_models]]"
      - "[[pomdp_framework]]"
      - "[[time_series_analysis]]"
---

# State Space Models

## Overview
A state space model describes a system through a latent state $x_t$ that changes over time and is observed only through measurements $y_t$. It separates two ingredients: a transition model for how the state evolves and an observation model for how measurements depend on the state. Many time-series methods, from navigation to neural data analysis, are instances of this structure.

## Formulation
In discrete time a general model is
$$x_{t+1}=f(x_t,u_t)+w_t,\qquad y_t=g(x_t)+v_t,$$
with process noise $w_t$, observation noise $v_t$ and optional inputs $u_t$. Probabilistically, the state is Markov, $p(x_{t+1}\mid x_{0:t})=p(x_{t+1}\mid x_t)$, and each observation depends only on the current state, so the joint distribution factorises as $p(x_0)\prod_t p(x_{t+1}\mid x_t)\,p(y_t\mid x_t)$.

Typical inference tasks are filtering ($p(x_t\mid y_{1:t})$), smoothing ($p(x_t\mid y_{1:T})$) and prediction. Two classic special cases are:
- **Linear Gaussian**: $f$ and $g$ are linear and noise is Gaussian, so filtering distributions remain Gaussian and are computed exactly by the Kalman filter.
- **Hidden Markov models**: the state is discrete, and filtering and smoothing are done with the forward-backward algorithm.

Nonlinear or non-Gaussian cases require approximations such as extended and unscented Kalman filters, particle filters or variational methods.

## Relation to Active Inference
Active Inference generative models are state space models with the addition of action. The discrete-time formulation is a partially observable Markov decision process in which the likelihood and transition matrices define the observation and transition models, and the transitions depend on control states. The continuous-time formulation uses stochastic differential equations for hidden states and causes, with inference in generalised coordinates of motion. Under linear Gaussian assumptions, the filtering performed by predictive coding schemes is closely related to Kalman-Bucy filtering. The distinctive feature of Active Inference is not the state space model itself, which is standard, but the use of the same variational free energy to perform inference and to select actions that bring about preferred observations.

## Related concepts
- [[hidden_states]] — the latent variables of the model
- [[state_estimation]] — filtering and smoothing the hidden state
- [[observation_model]] — the mapping from state to measurements
- [[transition_model]] — the dynamics of the hidden state
- [[markov_models]] — models sharing the Markov assumption
- [[pomdp_framework]] — the controlled discrete-state version used in Active Inference

## References
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. Journal of Basic Engineering, 82, 35-45.
- Sarkka, S. (2013). Bayesian Filtering and Smoothing. Cambridge University Press.
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
