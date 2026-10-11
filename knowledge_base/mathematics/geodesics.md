---
title: Geodesics
description: Curves that locally extremise length on a curved space, defined by the geodesic equation, with the Fisher-Rao geodesics of statistical manifolds and their role in geometric views of inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - differential_geometry
  - information_geometry
  - riemannian_geometry
semantic_relations:
  - type: relates
    links:
      - "[[differential_geometry]]"
      - "[[information_geometry]]"
      - "[[fisher_metric]]"
      - "[[statistical_manifolds]]"
      - "[[natural_gradients]]"
---

# Geodesics

## Overview
A geodesic generalises a straight line to a curved space. On a manifold with a metric $g$, a geodesic is a curve that is locally the shortest path between nearby points, and equivalently a curve with no intrinsic acceleration. On a sphere the geodesics are great circles, and in flat space they are straight lines.

## Definition
In local coordinates $x^k(t)$ a geodesic, parametrised proportionally to arc length, satisfies
$$\ddot x^k + \Gamma^k_{ij}\,\dot x^i \dot x^j = 0,$$
where $\Gamma^k_{ij}$ are the Christoffel symbols of the Levi-Civita connection built from $g$. The same curves extremise the length functional $\int\sqrt{g_{ij}\dot x^i\dot x^j}\,dt$ and the energy functional $\tfrac12\int g_{ij}\dot x^i\dot x^j\,dt$, so they arise from a variational principle (see [[variational_calculus]]). Geodesics are also the free-motion trajectories of mechanics on a manifold. Locally a geodesic minimises length, but globally it need not: on a sphere, the longer arc of a great circle is still a geodesic.

## Statistical manifolds
A family of probability distributions $p(x\mid\theta)$ forms a manifold whose natural metric is the Fisher information. For categorical distributions with probabilities $p_i$, the map $p_i\mapsto\sqrt{p_i}$ sends the probability simplex to part of a unit sphere, and the Fisher-Rao distance between $p$ and $q$ is $2\arccos\sum_i\sqrt{p_i q_i}$. Geodesics are then arcs of great circles. Information geometry also uses pairs of dual affine connections, whose "straight lines" (exponential and mixture geodesics) are generally not geodesics of the Fisher metric. Natural gradient descent follows the direction of steepest descent with respect to this metric, which is related to, but not the same as, travelling along a geodesic.

## Relation to Active Inference
Geometric treatments of the free energy principle view belief updating as motion on a statistical manifold, with the Fisher metric defining distances between beliefs and gradient flows on free energy defined relative to it. Geodesics describe the shortest paths between beliefs and the lengths of belief trajectories. This is an active research programme in geometric formulations of Active Inference and is not required for the standard discrete-state or continuous-time schemes.

## Related concepts
- [[fisher_metric]] — the metric whose geodesics are discussed
- [[information_geometry]] — geometry of statistical models
- [[natural_gradients]] — steepest descent in the Fisher metric
- [[geodesic_flows]] — the flow along geodesics on the tangent bundle
- [[geometric_active_inference]] — geometric formulations of belief updating
- [[differential_geometry]] — the general theory of curved spaces

## References
- do Carmo, M. P. (1992). Riemannian Geometry. Birkhauser.
- Amari, S., & Nagaoka, H. (2000). Methods of Information Geometry. American Mathematical Society and Oxford University Press.
