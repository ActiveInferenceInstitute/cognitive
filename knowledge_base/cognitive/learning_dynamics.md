---
title: Learning Dynamics
description: Learning dynamics describes how an agent's parameters and beliefs change over time during learning, including timescale separation, learning-rate effects, and Dirichlet count updating in Active Inference.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - learning
  - dynamics
  - parameters
semantic_relations:
  - type: relates
    links:
      - "[[learning_theory]]"
      - "[[learning_curves]]"
      - "[[stability_plasticity]]"
---

# Learning Dynamics

## Overview
Learning dynamics is the time course of change in a learner's parameters, beliefs, and performance as experience accumulates. It asks how fast learning proceeds, what it converges to, whether it is stable, and how it interacts with faster processes such as inference. The sense used here is the one cited from mathematical pages on learning rate and precision: parameter updating in a generative-model agent, plus its behavioural signature in learning curves.

## Formalism
For parameters $\theta$ and an objective $F$ such as variational free energy, continuous-time learning can be written as a gradient flow, $\dot\theta = -\kappa\,\partial F/\partial\theta$, where $\kappa$ is the learning rate. Discretized, this is the familiar update with step size $\kappa$. Key properties:

- **Timescale separation**: states are inferred fast, parameters change slowly, and hyperparameters (such as precisions) change slowest. Slow variables see time-averaged statistics of fast ones.
- **Learning rate trade-off**: large rates adapt quickly but are noisy and can be unstable; small rates are stable but slow. Annealed or count-based rates offer a compromise.
- **Count-based updating**: in discrete-state models, Dirichlet concentration parameters $a$ accumulate evidence, $a \leftarrow a + \eta\,(o \otimes s)$. The expected likelihood is $a/\sum a$, so after $n$ observations the effective step per observation shrinks roughly as $1/(a_0+n)$. Early data move beliefs far; late data move them little, a Bayesian analogue of a decaying learning rate. Without forgetting, the model becomes increasingly resistant to change in non-stationary worlds.
- **Learning curves**: performance typically improves rapidly and then plateaus, often summarized by a power law of practice (see [[learning_curves]]).

## Role in Active Inference
Learning is part of the same free-energy minimization as perception and action, operating on slower variables. Learning dynamics therefore determine how the epistemic term of [[expected_free_energy]] changes: as the agent's parameters become certain, the expected information gain from further exploration falls, and behaviour shifts toward exploiting preferences. Precision and temperature parameters ([[precision_parameter]], [[temperature_parameter]]) interact with learning, since they scale how strongly errors and policies influence updates. Which learning rule biological systems use is open; the gradient and Dirichlet forms are modelling choices consistent with the framework.

## Related concepts
- [[learning_theory]] — the broader theory of how learning occurs
- [[learning_curves]] — behavioural trajectories of performance over practice
- [[stability_plasticity]] — the trade-off that learning rate controls
- [[gradient_descent]] — the optimization process underlying many updates
- [[synaptic_plasticity]] — biological substrate for parameter change
- [[expected_free_energy_update]] — how learning enters policy evaluation
- [[exploration_exploitation]] — balance that shifts as learning proceeds
