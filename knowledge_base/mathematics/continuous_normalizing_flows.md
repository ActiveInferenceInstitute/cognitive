---
title: Continuous Normalizing Flows
description: Continuous normalizing flows define invertible transformations of probability densities through neural ordinary differential equations, with the log-density changing by the integrated divergence of a learned vector field.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - generative_models
  - normalizing_flows
  - differential_equations
semantic_relations:
  - type: relates
    links:
      - "[[normalizing_flows]]"
      - "[[neural_odes]]"
      - "[[differential_equations]]"
      - "[[deep_generative_models]]"
      - "[[variational_inference]]"
---

# Continuous Normalizing Flows

## Overview
A normalizing flow transforms a simple base density, such as a Gaussian, into a complex target density by a sequence of invertible maps. A continuous normalizing flow (CNF) replaces the discrete sequence with the flow of an ordinary differential equation whose right-hand side is a neural network. It was introduced as an application of neural ordinary differential equations by Chen and colleagues.

## Formulation
Let $z(t)$ solve
$$\frac{dz}{dt}=f_\theta(z(t),t),$$
with $z(t_0)$ drawn from the base density. The map from $z(t_0)$ to $z(t_1)$ is invertible whenever $f_\theta$ is Lipschitz in $z$, since solutions of such an equation are unique and can be integrated backwards. The density evolves according to the instantaneous change of variables formula
$$\frac{\partial \log p(z(t))}{\partial t}=-\operatorname{tr}\!\left(\frac{\partial f_\theta}{\partial z}\right),$$
so that
$$\log p(z(t_1))=\log p(z(t_0))-\int_{t_0}^{t_1}\operatorname{tr}\!\left(\frac{\partial f_\theta}{\partial z}\right)dt.$$
This replaces the Jacobian determinant of a discrete flow, which costs $O(D^3)$ for dimension $D$ in general, with a trace, so the architecture of $f_\theta$ need not be restricted to triangular Jacobians. Gradients with respect to $\theta$ are computed by solving an adjoint ODE backwards in time, with memory that does not grow with the number of solver steps. The FFJORD model estimates the trace stochastically with Hutchinson's estimator, which gives an unbiased estimate at roughly the cost of one vector-Jacobian product.

## Training and variants
Training by maximum likelihood requires numerical integration at every step, which is costly. Flow-matching methods avoid simulation by regressing the vector field onto a prescribed probability path between the base and data distributions. The deterministic probability-flow ODE associated with a diffusion model is also a continuous normalizing flow, linking the two families.

## Relation to Active Inference
Flexible variational posteriors built from flows are one way to relax a mean-field or Gaussian approximation in variational inference, and a CNF can serve as such a posterior or as a generative model of observations. This use appears in machine-learning implementations of deep active inference. It is a tool for representing densities and is not part of the core theory.

## Related concepts
- [[normalizing_flows]] — the discrete-time precursor of the construction
- [[neural_odes]] — the continuous-depth networks that supply the vector field
- [[differential_equations]] — the mathematical setting
- [[diffusion_models]] — models whose probability-flow ODE is a CNF
- [[taylor_series]] — expansions used in ODE solvers and jet-based approximations
- [[variational_inference]] — a main use of flexible densities

## References
- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. Advances in Neural Information Processing Systems, 31.
- Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2019). FFJORD: Free-form continuous dynamics for scalable reversible generative models. International Conference on Learning Representations.
- Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows. Proceedings of the 32nd International Conference on Machine Learning, 1530-1538.
