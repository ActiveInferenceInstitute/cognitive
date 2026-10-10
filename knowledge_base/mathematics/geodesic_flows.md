---
title: Geodesic Flows
description: The flow on the tangent bundle of a Riemannian manifold that moves points along geodesics, with the geodesic equation, Fisher-Rao examples, and a clear contrast with free-energy gradient flows.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - differential_geometry
  - information_geometry
  - dynamics
semantic_relations:
  - type: relates
    links:
      - "[[differential_geometry]]"
      - "[[information_geometry]]"
      - "[[statistical_manifolds]]"
      - "[[geometric_mechanics]]"
---

# Geodesic Flows

## Overview
On a Riemannian manifold $(M,g)$ a geodesic is a curve that is locally length-minimising and has zero acceleration with respect to the Levi-Civita connection. The geodesic flow is the dynamical system on the tangent bundle $TM$ that sends a point and velocity $(x,v)$ to the position and velocity reached after time $t$ along the geodesic with those initial data. Because a geodesic is a free particle on the manifold, the flow preserves kinetic energy $\tfrac12 g(v,v)$ and is a Hamiltonian flow on $T^*M$ with $H=\tfrac12 g^{ij}p_ip_j$.

## Formalism
In local coordinates the geodesic equation is
$$\ddot x^k+\Gamma^k_{ij}(x)\,\dot x^i\dot x^j=0,$$
with Christoffel symbols $\Gamma$ of the metric. Existence and uniqueness follow from ODE theory. The exponential map $\exp_x(v)$ evaluates the flow at time 1.

**Information-geometric examples.** On a statistical manifold with the Fisher metric ([[information_geometry]]):
- For the univariate Gaussian family with parameters $(\mu,\sigma)$ the Fisher metric is $ds^2=(d\mu^2+2\,d\sigma^2)/\sigma^2$, a rescaled hyperbolic plane, so geodesics are semicircles centred on $\sigma=0$ in suitable coordinates.
- For categorical distributions, the square-root map $p\mapsto 2\sqrt p$ embeds the simplex into a sphere, and the Fisher-Rao distance is $2\arccos\sum_i\sqrt{p_iq_i}$.

## Contrast with gradient flows
Geodesic flow is second order and conservative: it carries inertia and conserves energy. Free-energy minimisation is first order and dissipative: $\dot x=-g^{-1}\nabla F$ ([[gradient_flows]]). Descent along the natural gradient follows the steepest-descent direction in the Fisher metric but its trajectories are not geodesics. Mixing them appears in second-order optimisers with momentum, whose continuous limit adds an inertial term and damping.

## Role in Active Inference
Geodesic ideas enter through the geometry of belief space: distances between beliefs are measured with the Fisher-Rao metric, and natural-gradient updates make belief updating invariant to parameterisation. The geometric Active Inference programme uses these structures ([[geometric_active_inference]]). Claims that belief updating or optimal policies follow geodesics are informal unless a specific action functional is given; for a stationary density with solenoidal flow, motion along level sets of surprise is closer to Hamiltonian than to geodesic motion. The page therefore describes a geometric tool, not a result of the free-energy principle.

## Related concepts
- [[differential_geometry]] — connections, curvature and the geodesic equation
- [[information_geometry]] — Fisher metric on distribution families
- [[natural_gradients]] — steepest descent in the Fisher metric
- [[gradient_flows]] — dissipative alternative to geodesic motion
- [[geodesic_geometry]] — broader use of geodesics, including Fuller's sense
- [[hamiltonian_mechanics]] — framework in which geodesic flow is Hamiltonian
