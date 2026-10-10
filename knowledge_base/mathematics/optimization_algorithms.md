---
title: Optimization Algorithms
description: "Overview of iterative optimization algorithms, with gradient descent rates and natural gradient preconditioning, and their role in minimising variational free energy."
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - optimization
  - gradient_descent
  - convergence
aliases:
  - computational_optimization
  - numerical_optimization
  - optimization
  - optimization_techniques
semantic_relations:
  - type: foundation
    links:
      - "[[optimization_theory]]"
      - "[[calculus]]"
  - type: relates
    links:
      - "[[natural_gradients]]"
      - "[[stochastic_optimization]]"
      - "[[variational_inference]]"
---

# Optimization Algorithms

## Overview
Optimization algorithms are iterative procedures that search for a minimiser of an objective $f$ over a feasible set. They differ in what they assume about $f$ (smooth, convex, noisy, black-box) and in what information they use (function values, gradients, curvature).

## Definition / Formalism
Gradient descent iterates $x_{k+1}=x_k-\eta\nabla f(x_k)$. If $f$ is $L$-smooth and $\mu$-strongly convex and $\eta=1/L$, then

$$f(x_k)-f^*\le\Big(1-\tfrac{\mu}{L}\Big)^k\,(f(x_0)-f^*),$$

so the condition number $L/\mu$ controls the rate. For the quadratic $f=\tfrac12(x_1^2+10x_2^2)$ ($\mu=1$, $L=10$, $\eta=0.1$) the bound factor is $0.9^k$; the actual iterate norm after 20 steps is about 0.086 of its start, against $0.9^{20}\approx0.12$. Newton's method uses curvature, $x_{k+1}=x_k-H(x_k)^{-1}\nabla f(x_k)$, and solves a strictly convex quadratic in one step. Quasi-Newton methods (BFGS) approximate $H^{-1}$ from gradient differences. Stochastic gradient methods replace $\nabla f$ with an unbiased noisy estimate and need decaying or small step sizes. Coordinate methods update one block at a time; coordinate ascent variational inference and expectation-maximisation are of this kind, each step increasing a lower bound.

Natural gradient descent preconditions by the Fisher information matrix, $x_{k+1}=x_k-\eta F(x_k)^{-1}\nabla f(x_k)$, which makes the step invariant to reparametrisation of a probability model.

## Role in Active Inference
Variational free energy is minimised by exactly these tools. Perception is gradient descent (in the continuous-time agent, a gradient flow on $F$ over generalized coordinates; `ContinuousActiveInference.step` in `code/Things/Continuous_Generic/continuous_generic.py` applies exact gradient steps with a step size `alpha`). Discrete agents use fixed-point or message-passing updates that are coordinate-type iterations. Learning of model parameters is a slower gradient scheme on the same functional. The strength of the link is high: this is how the machinery runs. What the theory does not provide is a convergence guarantee in general, since $F$ is non-convex for most generative models and results depend on initialisation.

## Related concepts
- [[optimization_theory]] — convexity, duality and optimality conditions behind these algorithms
- [[gradient_descent]] — the free-energy gradient-flow view in this vault
- [[natural_gradients]] — information-geometric preconditioning
- [[stochastic_optimization]] — noisy-gradient and annealing methods
- [[automatic_differentiation]] — how gradients are computed in practice
- [[convergence_analysis]] — rates and stopping criteria
- [[trajectory_optimization]] — optimisation over time-indexed decision variables
