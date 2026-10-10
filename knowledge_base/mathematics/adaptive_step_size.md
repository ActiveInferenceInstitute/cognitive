---
title: Adaptive Step Size
description: Strategies that adjust the step length of numerical integrators and gradient-based optimisers from local error or curvature estimates, with relevance to Taylor-series integration and free-energy descent.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - numerical_methods
  - optimization
  - integration
semantic_relations:
  - type: relates
    links:
      - "[[numerical_methods]]"
      - "[[taylor_series]]"
      - "[[numerical_stability]]"
      - "[[optimization_algorithms]]"
---

# Adaptive Step Size

## Overview
Step-size adaptation chooses the increment $h$ of an iterative scheme at run time instead of fixing it in advance. It applies in two settings that share the idea: numerical integration of differential equations, where $h$ is a time increment, and optimisation, where $h$ is a learning rate or line-search length. The aim is to take large steps where the solution is smooth and small ones where it changes quickly, keeping error or progress under control.

## Mechanisms
**Embedded Runge-Kutta pairs.** Two methods of orders $p$ and $p+1$ share function evaluations. Their difference gives a local error estimate $e$. A standard controller proposes
$$h_{\text{new}} = h\,\min\!\left(f_{\max},\ \max\!\left(f_{\min},\ 0.9\,(\mathrm{tol}/e)^{1/(p+1)}\right)\right),$$
and a step is rejected and retried if $e>\mathrm{tol}$. Dormand-Prince 5(4) is a common example.

**Taylor-series integrators.** A truncated Taylor expansion of order $n$ with coefficients $a_k$ has leading neglected term of size about $|a_n|h^n$. Requiring this to stay below a tolerance $\varepsilon$ gives a step $h\approx(\varepsilon/|a_n|)^{1/n}$, so the step shrinks automatically near singularities or fast transients. See [[taylor_series]] and [[numerical_taylor_methods]].

**Stiff problems.** Explicit steps are limited by stability rather than accuracy, so a controller keeps $h$ small even when error is tiny; implicit methods avoid this at the cost of solving equations each step ([[numerical_stability]]).

**Optimisation.** Backtracking line search shrinks $h$ until a sufficient-decrease condition (Armijo) holds. Adaptive learning-rate methods rescale per-coordinate steps using gradient statistics. Natural-gradient steps use the Fisher metric to set direction and scale ([[natural_gradients]]).

## Role in Active Inference
Implementations of Active Inference integrate or iterate on free energy in two places. Continuous-time schemes (generalised filtering, DEM) integrate gradient flows on variational free energy in generalised coordinates, and discrete-time schemes iterate belief updates until the free energy stops changing. Both can use a fixed step in practice. Adaptive control is an implementation choice that improves stability and cost, not part of the theory; the free-energy principle itself says nothing about the integrator. The link is practical: poor step choice can make a simulated free-energy descent diverge or oscillate, which would be wrongly read as a property of the model.

## Related concepts
- [[numerical_methods]] — general framework for approximate solution of continuous problems
- [[taylor_series]] — expansion whose truncation error drives the step rule
- [[numerical_stability]] — stability limits that bound admissible steps
- [[optimization_algorithms]] — learning-rate and line-search variants
- [[gradient_flows]] — the continuous dynamics that these steps discretise
- [[convergence_control]] — stopping and tolerance logic that works alongside step control
