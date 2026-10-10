---
title: Gradient Flows
description: Continuous-time dynamics that descend a potential or functional along the steepest direction of a chosen metric, covering Euclidean, Riemannian and Wasserstein cases and their role in free-energy minimisation.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - dynamical_systems
  - optimization
  - information_geometry
semantic_relations:
  - type: relates
    links:
      - "[[natural_gradients]]"
      - "[[optimal_transport]]"
      - "[[fokker_planck]]"
      - "[[dynamical_systems]]"
---

# Gradient Flows

## Overview
A gradient flow is the evolution $\dot x=-\nabla F(x)$ of a state that moves downhill on a function $F$ as fast as possible. The result depends on the metric used to turn the differential $dF$ into a direction, so the same $F$ can generate several different flows. Along any gradient flow $F$ decreases, $\tfrac{d}{dt}F=-|\nabla F|^2\le0$, so the flow is dissipative and has no periodic orbits other than fixed points.

## Variants
- **Euclidean:** $\dot x=-\nabla F(x)$. Explicit Euler discretisation is gradient descent with step $h$ ([[adaptive_step_size]]).
- **Riemannian:** $\dot x=-G(x)^{-1}\nabla F(x)$ for metric $G$. With the Fisher information as $G$ this is the natural gradient flow of Amari ([[natural_gradients]]), which is invariant under reparameterisation of the distribution family.
- **Wasserstein:** In the space of probability densities with the 2-Wasserstein metric, the gradient flow of $\mathrm{KL}(\rho\Vert e^{-V})$ is exactly the Fokker-Planck equation $\partial_t\rho=\nabla\cdot(\rho\nabla V)+\nabla^2\rho$ (Jordan, Kinderlehrer and Otto, 1998). Diffusion is therefore steepest descent of free energy in transport geometry ([[optimal_transport]]).
- **Gradient flows with a skew part:** $\dot x=-(\Gamma+Q)\nabla F$, with $\Gamma$ symmetric positive and $Q$ antisymmetric, keep $F$ non-increasing because the $Q$ term contributes zero to $\dot F$. These are the generalised gradient flows relevant in physics.
- **Convergence.** For $\lambda$-convex $F$ the flow contracts exponentially at rate $\lambda$, and the Łojasiewicz inequality governs convergence for non-convex analytic $F$.

## Role in Active Inference
Perception in Active Inference is a gradient flow on variational free energy with respect to the sufficient statistics of beliefs, written $\dot\mu=-\partial_\mu F$ (plus a term for motion in generalised coordinates), and action is a gradient flow of $F$ with respect to the action variable. At the level of the principle, the dynamics at non-equilibrium steady state have the form $f=(Q-\Gamma)\nabla\ln p^*$, i.e. a gradient flow on surprise $-\ln p^*$ (with metric $\Gamma$) combined with a solenoidal term. Which metric is used is a modelling choice with real consequences: Euclidean flows are parameterisation-dependent, whereas natural-gradient flows are not. Implementations discretise the flow, so step size and integrator choice affect results.

## Related concepts
- [[natural_gradients]] — gradient flow in the Fisher metric
- [[fokker_planck]] — the density flow that is a Wasserstein gradient flow
- [[geodesic_flows]] — conservative counterpart to dissipative gradient flow
- [[variational_free_energy]] — the functional descended in perception
- [[optimization_theory]] — convergence theory for descent methods
- [[non_equilibrium_steady_state]] — setting in which flows have gradient and solenoidal parts

## References
- Amari, S. (1998). Natural gradient works efficiently in learning. *Neural Computation*, 10, 251-276.
- Jordan, R., Kinderlehrer, D. and Otto, F. (1998). The variational formulation of the Fokker-Planck equation. *SIAM Journal on Mathematical Analysis*, 29, 1-17.
