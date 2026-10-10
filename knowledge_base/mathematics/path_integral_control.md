---
title: Path Integral Control
description: "Kappen's class of stochastic optimal control problems that linearise via a log transform, so the optimal cost-to-go is an expectation over uncontrolled paths, and its relation to KL control and Active Inference."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - control
  - stochastic_control
  - path_integral
semantic_relations:
  - type: relates
    links:
      - "[[path_integral_free_energy]]"
      - "[[optimal_control]]"
      - "[[stochastic_differential_equations]]"
      - "[[action_selection]]"
---

# Path Integral Control

## Overview
Path integral control is a class of stochastic optimal control problems, introduced by Kappen (2005), in which the Hamilton-Jacobi-Bellman (HJB) equation becomes linear after a logarithmic change of variables. The optimal cost-to-go can then be estimated by sampling trajectories of the uncontrolled system, without solving a nonlinear PDE.

## Formalism
Consider $dx=f(x,t)\,dt+g(x,t)\,(u\,dt+d\xi)$ with noise covariance $\nu\,dt$ and cost
$$J=\mathbb E\Big[\phi(x_T)+\int_t^T\big(V(x_s)+\tfrac12u^\top Ru\big)ds\Big].$$
Noise and control act through the same channel $g$, and the control cost satisfies $R=\lambda\nu^{-1}$ for some $\lambda>0$. Under this condition, define the desirability $\Psi=\exp(-J/\lambda)$ (Cole-Hopf transform). The HJB equation becomes linear in $\Psi$, and
$$\Psi(x,t)=\mathbb E_{\text{uncontrolled}}\Big[\exp\!\big(-\tfrac1\lambda S(\tau)\big)\Big],\quad S=\phi(x_T)+\int V\,ds,$$
an average over paths, hence the name. The optimal control is $u^*=-R^{-1}g^\top\partial_xJ$.

**Discrete analogue (KL control).** Todorov's linearly solvable MDPs minimise $D_{KL}(u\,\|\,p)+\mathbb E_u[v(x')]$ over controlled transition distributions. The minimiser is $u\propto p(x'\mid x)e^{-v(x')}$ with optimal value $-\ln\sum_{x'}p(x'\mid x)e^{-v(x')}$; I checked this numerically on a random five-state chain. Path-integral control is the continuous-time counterpart.

**Sampling methods.** Policy Improvement with Path Integrals (PI$^2$, Theodorou, Buchli and Schaal 2010) reweights sampled trajectories by $\exp(-S/\lambda)$ and averages their noise to update parameters of a controller.

## Role in Active Inference
Both frameworks recast control as inference, and the structure is shared: the control penalty is a KL divergence from passive dynamics, and the solution is a reweighting of prior paths by an exponentiated cost. The correspondence is exact for the stated class, but Active Inference does not generally reduce to it: its objective is [[expected_free_energy]], whose risk term is a KL divergence to preferred outcomes and whose ambiguity term has no counterpart in a pure cost problem. Treatments in [[path_integral_free_energy]] and [[path_integral_bridge]] pursue this link, and it should be read as a research programme rather than a settled equivalence.

## Related concepts
- [[path_integral]] — the underlying path-space formalism
- [[optimal_control]] — HJB equations and dynamic programming
- [[ito_calculus]] — stochastic calculus used in the derivation
- [[value_function]] — the cost-to-go $J$ is minus a value function
- [[action_selection]] — where control enters Active Inference

## References
- Kappen, H. J. (2005). Linear theory for control of nonlinear stochastic systems. Physical Review Letters 95, 200201.
- Todorov, E. (2009). Efficient computation of optimal actions. Proceedings of the National Academy of Sciences.
- Theodorou, E., Buchli, J., Schaal, S. (2010). A generalized path integral control approach to reinforcement learning. Journal of Machine Learning Research.
