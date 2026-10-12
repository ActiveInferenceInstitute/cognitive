---
title: Christoffel Symbols
description: Christoffel symbols are the coefficients of the Levi-Civita connection on a Riemannian manifold, computed from the metric and its derivatives, and they define covariant derivatives and geodesic equations.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - differential_geometry
  - riemannian_geometry
  - information_geometry
semantic_relations:
  - type: relates
    links:
      - "[[riemannian_metric]]"
      - "[[metric_tensor]]"
      - "[[geodesic_equations]]"
      - "[[differential_geometry]]"
      - "[[fisher_rao_metric]]"
---

# Christoffel Symbols

## Overview
Christoffel symbols $\Gamma^k_{ij}$ are the coefficients that express how basis vectors change from point to point in a chosen coordinate system on a manifold with a metric. They are needed because partial derivatives of a vector field do not transform as a tensor under change of coordinates, whereas the covariant derivative, built with the symbols, does.

## Definition
For a metric $g_{ij}$ with inverse $g^{ij}$, the Christoffel symbols of the Levi-Civita connection are
$$\Gamma^k_{ij}=\tfrac12\, g^{kl}\left(\partial_i g_{jl}+\partial_j g_{il}-\partial_l g_{ij}\right),$$
with summation over repeated indices. The Levi-Civita connection is the unique connection that is torsion-free, which gives $\Gamma^k_{ij}=\Gamma^k_{ji}$, and compatible with the metric, so that parallel transport preserves lengths and angles. The symbols are not the components of a tensor: they can be made to vanish at a single point by a suitable choice of coordinates, but not in a neighbourhood unless the manifold is flat.

## Uses
The covariant derivative of a vector field is $\nabla_i V^k=\partial_i V^k+\Gamma^k_{ij}V^j$. A curve $x^k(\tau)$ is a geodesic when its acceleration is zero in this sense:
$$\ddot x^k+\Gamma^k_{ij}\,\dot x^i\dot x^j=0.$$
The Riemann curvature tensor is assembled from first derivatives of $\Gamma$ and products of two symbols, so curvature is determined by the metric alone. In polar coordinates on the plane the nonzero symbols are $\Gamma^r_{\theta\theta}=-r$ and $\Gamma^\theta_{r\theta}=\Gamma^\theta_{\theta r}=1/r$, and these describe the coordinate system, not any curvature.

## Relation to Active Inference
In information geometry the Fisher-Rao metric turns a family of probability distributions into a Riemannian manifold, and the Christoffel symbols of that metric give the geodesics along which belief updates can be described. For the univariate Gaussian family with parameters $(\mu,\sigma)$ the Fisher metric is $ds^2=(d\mu^2+2\,d\sigma^2)/\sigma^2$, a space of constant negative curvature $-1/2$. Natural-gradient descent, used in variational inference, relies on this metric. The relation to Active Inference is a mathematical tool for describing belief dynamics geometrically; it is used in geometric treatments of the free energy principle, and is not a claim about how brains compute.

## Related concepts
- [[riemannian_metric]] — the structure from which the symbols are computed
- [[geodesic_equations]] — the equations in which the symbols appear
- [[differential_geometry]] — the field in which connections are studied
- [[fisher_rao_metric]] — the metric on spaces of probability distributions
- [[natural_gradients]] — optimisation that uses the Fisher metric
- [[holonomy]] — the global effect of parallel transport defined by the connection

## References
- do Carmo, M. P. (1992). Riemannian Geometry. Birkhauser.
- Lee, J. M. (2018). Introduction to Riemannian Manifolds (2nd ed.). Springer.
- Amari, S., & Nagaoka, H. (2000). Methods of Information Geometry. American Mathematical Society and Oxford University Press.
