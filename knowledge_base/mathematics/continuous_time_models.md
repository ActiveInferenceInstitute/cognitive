---
title: Continuous-Time Models
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - continuous_time
  - generalized_coordinates
  - dynamical_models
semantic_relations:
  - type: foundation
    links:
      - "[[differential_equations]]"
      - "[[dynamical_systems]]"
  - type: implements
    links:
      - "[[generalized_coordinates]]"
      - "[[stochastic_differential_equations]]"
      - "[[continuous_discrete_inference]]"
---

# Continuous-Time Models

## Overview
A continuous-time model describes a system's state as a function of time $t\in\mathbb R$ through differential equations, rather than as a sequence of discrete steps. In active inference it is the formulation used for movement, sensorimotor control and other signals that are naturally smooth in time; the discrete-time POMDP is the alternative for categorical decisions.

## Definition / Formalism
A generative model in continuous time pairs a flow with an observation map and noise:

$$\dot x=f(x,v,\theta)+w,\qquad y=g(x,v,\theta)+z,$$

with hidden states $x$, exogenous causes $v$, parameters $\theta$ and random fluctuations $w,z$. Because real fluctuations are smooth rather than white, the model is written in generalized coordinates $\tilde x=(x,x',x'',\dots)$, with a shift operator $D$ that maps each order to the next, $D\tilde x=(x',x'',\dots)$. The model predicts both the motion and its derivatives, $D\tilde x=\tilde f(\tilde x,\tilde v)+\tilde w$. Under the Laplace approximation, state estimation (dynamic expectation maximisation) takes the form

$$\dot{\tilde\mu}=D\tilde\mu-\partial_{\tilde\mu}F,\qquad \dot a=-\partial_aF,$$

where the first term transports the estimate along its own predicted motion and the second descends the free-energy gradient; at a minimum the estimate follows the true trajectory, $\dot{\tilde\mu}=D\tilde\mu$. Simulation requires discretisation. For $\dot x=\lambda x$ with $\lambda<0$, the explicit Euler map $x_{k+1}=(1+\lambda\Delta t)x_k$ is stable only for $\Delta t<2/|\lambda|$; with $\lambda=-10$, $\Delta t=0.19$ decays and $\Delta t=0.21$ diverges. Stiff dynamics therefore call for smaller steps or implicit and local-linearisation integrators.

## Role in Active Inference
This is where continuous-state perception and action are formulated: perception is a gradient flow on free energy in generalized coordinates, and action suppresses proprioceptive prediction error. The vault's continuous-time note builds on it, and `ContinuousActiveInference` in `code/Things/Continuous_Generic/continuous_generic.py` implements precision-weighted generalized-coordinate updates by gradient steps of size `alpha` over `dt`. Discrete and continuous models can be nested, with continuous dynamics at the lower level producing evidence for discrete categories above. Noise handling is covered in the stochastic differential equation page.

## Related concepts
- [[continuous_time_active_inference]] — cognitive and architectural treatment of this setting
- [[generalized_coordinates]] — temporal derivatives as an extended state
- [[stochastic_differential_equations]] — models with random fluctuations
- [[continuous_discrete_inference]] — combining continuous and discrete levels
- [[state_space]] — the state-space form of the dynamics
- [[predictive_coding]] — message-passing scheme implementing these updates

## References
- Friston, Stephan, Li & Daunizeau (2010), Generalised filtering, Mathematical Problems in Engineering.
- Buckley, Kim, McGregor & Seth (2017), The free energy principle for action and perception: a mathematical review, Journal of Mathematical Psychology 81.
