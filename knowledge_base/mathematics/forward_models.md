---
title: Forward Models
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - prediction
  - motor_control
  - generative_models
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/generative_models]]"
      - "[[knowledge_base/mathematics/pomdp_structure]]"
      - "[[knowledge_base/cognitive/motor_control]]"
      - "[[knowledge_base/cognitive/temporal_models]]"
---

# Forward Models

## Overview
A forward model predicts the sensory consequences of a state or action: it maps causes to effects. In control theory and motor neuroscience it is contrasted with an inverse model, which maps a desired outcome back to the action that would produce it. Forward models are used to predict feedback before it arrives, to cancel self-generated sensation, and to estimate state despite delays and noise.

## Definition / Formalism
A deterministic forward model is a pair of maps

$$x_{t+1}=f(x_t,u_t),\qquad \hat y_t=g(x_t),$$

with state $x$, action (motor command) $u$, and predicted sensation $\hat y$. A probabilistic version replaces the maps with conditionals $p(x_{t+1}\mid x_t,u_t)$ and $p(y_t\mid x_t)$. Combined with an observation, a predictive stage and a correction stage give a recursive state estimator; in the linear-Gaussian case this is the Kalman filter, whose prediction step is $\hat x_{t+1|t}=F\hat x_{t|t}+Gu_t$ and whose correction weights the prediction error $y_t-g(\hat x_{t|t-1})$ by a gain. The prediction error is the signal that drives learning of the model itself.

An inverse model computes $u=h(x_{\text{current}},x_{\text{desired}})$ and is generally ill-posed, because many commands can produce the same outcome.

Discrete example, using the repo's matrix convention: with column-stochastic $A$ and $B^{(a)}$, the forward prediction is $q(s')=B^{(a)}q(s)$ and $q(o)=A\,q(s')$. For $A=\begin{pmatrix}0.9&0.1\\0.1&0.9\end{pmatrix}$ and $q(s)=(0.7,0.3)$, identity transitions predict $q(o)=(0.66,0.34)$ and the swap-like $B^{(1)}=\begin{pmatrix}0.1&0.9\\0.9&0.1\end{pmatrix}$ predicts $(0.372,0.628)$.

## Role in Active Inference
In this repo, forward prediction is the first step of policy evaluation: `DiscreteGenerativeModel.predict_states` and `predict_observations` compute exactly the two products above, and expected free energy is then evaluated on the predicted observation distribution (risk against preferences, plus ambiguity). A policy is scored by rolling the forward model ahead. Action is selected by minimising expected free energy under predicted outcomes, with preferences encoded as prior beliefs, rather than by inverting the model into a command. In the motor-control formulation, descending signals are proposed to carry predictions of proprioceptive input that reflex arcs fulfil, which removes the need for an explicit inverse model. This is a theoretical proposal, not something this repository implements.

## Related concepts
- [[generative_models]] — forward models are the causal direction of a generative model
- [[pomdp_structure]] — $A$ and $B$ as observation and transition models
- [[motor_control]] — prediction of sensory feedback during movement
- [[temporal_models]] — predicting across time steps
- [[expected_free_energy]] — evaluated on forward-predicted outcomes
- [[predictive_coding]] — prediction errors as the update signal

## References
- Wolpert, D. M., Ghahramani, Z. and Jordan, M. I. (1995). An internal model for sensorimotor integration. *Science* 269, 1880-1882.
- Wolpert, D. M. and Kawato, M. (1998). Multiple paired forward and inverse models for motor control. *Neural Networks* 11, 1317-1329.
- Adams, R. A., Shipp, S. and Friston, K. J. (2013). Predictions not commands: active inference in the motor system. *Brain Structure and Function* 218, 611-643.
