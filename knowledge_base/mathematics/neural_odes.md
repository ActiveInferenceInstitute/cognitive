---
title: Neural ODEs
description: Neural networks whose hidden state evolves according to a learned ordinary differential equation, with adjoint-based training, continuous normalising flows, and their connection to continuous-time models in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - deep_learning
  - differential_equations
  - continuous_time
semantic_relations:
  - type: relates
    links:
      - "[[calculus]]"
      - "[[dynamical_systems]]"
      - "[[continuous_time_models]]"
      - "[[normalizing_flows]]"
      - "[[diffusion_models]]"
      - "[[deep_learning]]"
---

# Neural ODEs

## Overview
A neural ordinary differential equation replaces a stack of discrete layers with a continuous-time dynamical system whose vector field is a neural network. A residual network updates its hidden state as $h_{k+1}=h_k+f(h_k,\theta_k)$, which is one Euler step of an ODE. Taking the step size to zero gives
$$\frac{dh}{dt}=f_\theta\big(h(t),t\big),\qquad h(0)=\text{input},$$
and the output is $h(T)$, computed by a numerical solver such as a Runge-Kutta method with adaptive step size (Chen et al., 2018).

## Training and properties
Gradients are obtained with the adjoint sensitivity method. The adjoint state $a(t)=\partial L/\partial h(t)$ obeys a second ODE that is integrated backward in time, so memory cost does not grow with the number of solver steps. The solver's tolerance trades accuracy for computation, and the effective depth varies with the input. Because the dynamics are defined in continuous time, observations at irregular times can be handled naturally, which suits time-series data.

Neural ODEs also give continuous normalising flows. If the state is a sample from a density, its log-density changes along the trajectory by $\frac{d}{dt}\log p(h(t))=-\operatorname{tr}\big(\partial f/\partial h\big)$, avoiding the determinant of a discrete flow (see [[normalizing_flows]]). A limitation is that an ODE flow is a homeomorphism and cannot change the topology of the input space, so some maps need extra dimensions to represent.

## Relation to Active Inference
Continuous-time Active Inference models are written as differential equations for hidden states and causes, with beliefs updated by gradient descent on free energy in generalised coordinates (see [[continuous_time_models]]). A neural ODE could in principle parameterise the flow of such a generative model, and learning its parameters corresponds to model learning. This is a point of technical compatibility and research interest rather than part of the standard formulation. The link to [[dynamical_systems]] is direct, since a trained neural ODE is a dynamical system, but the two traditions use different objectives and training procedures.

## Related concepts
- [[calculus]] — the differential equations underlying the model
- [[dynamical_systems]] — the framework in which the learned vector field is analysed
- [[continuous_time_models]] — continuous-time generative models in Active Inference
- [[normalizing_flows]] — continuous flows as an extension of discrete ones
- [[diffusion_models]] — the stochastic counterpart through SDEs and the probability-flow ODE

## References
- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. Advances in Neural Information Processing Systems, 31, 6571-6583.
- Hairer, E., Norsett, S. P., & Wanner, G. (1993). Solving Ordinary Differential Equations I: Nonstiff Problems (2nd ed.). Springer.
