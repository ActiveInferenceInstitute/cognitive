---
title: Metric Tensor
description: The smoothly varying inner product on tangent spaces that defines lengths, angles and volumes on a manifold, with coordinate transformation, standard examples and its use in information geometry.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - differential_geometry
  - riemannian_geometry
  - tensors
semantic_relations:
  - type: relates
    links:
      - "[[riemannian_metric]]"
      - "[[differential_geometry]]"
      - "[[geodesic_equations]]"
      - "[[inner_products]]"
      - "[[fisher_rao_metric]]"
wikidata: Q757269
---

# Metric Tensor

## Overview
A metric tensor $g$ on a smooth manifold $M$ assigns to each point $p$ an inner product $g_p$ on the tangent space $T_pM$, depending smoothly on $p$. In local coordinates $x^i$ it is a symmetric, positive-definite matrix of functions $g_{ij}(x)$, and the infinitesimal squared length is
$$ds^2=g_{ij}(x)\,dx^i\,dx^j.$$
A manifold equipped with such a $g$ is a Riemannian manifold. If positivity is relaxed to non-degeneracy with indefinite signature, as in special and general relativity, the structure is pseudo-Riemannian.

## What it determines
- **Length of curves:** $L[\gamma]=\int\sqrt{g_{ij}\dot x^i\dot x^j}\,dt$, and hence the distance between points as the infimum of lengths.
- **Angles:** the angle between tangent vectors $u,v$ satisfies $\cos\theta=g(u,v)/\sqrt{g(u,u)g(v,v)}$.
- **Volume:** the Riemannian volume element is $\sqrt{\det g}\,dx^1\cdots dx^n$.
- **Index raising:** the inverse $g^{ij}$ converts covectors to vectors, so that the gradient of a function is $(\nabla f)^i=g^{ij}\partial_jf$. The gradient therefore depends on the metric, which is why steepest descent changes when the metric changes.
- **Connection and curvature:** the metric fixes a unique torsion-free, metric-compatible connection, the Levi-Civita connection, and from it the Christoffel symbols, geodesics and curvature.

Under a change of coordinates $x\mapsto x'$, the components transform as
$$g'_{ab}=\frac{\partial x^i}{\partial x'^a}\frac{\partial x^j}{\partial x'^b}\,g_{ij},$$
which is the rule for a covariant tensor of rank two.

## Examples
In the Euclidean plane, $g_{ij}=\delta_{ij}$ in Cartesian coordinates, and in polar coordinates $ds^2=dr^2+r^2d\theta^2$. On a sphere of radius $R$, $ds^2=R^2(d\theta^2+\sin^2\theta\,d\phi^2)$. On a statistical manifold the Fisher information matrix is the canonical choice.

## Relation to Active Inference
The link is geometric and active in the literature. The Fisher metric on belief space defines the natural gradient, which gives parametrisation-independent descent on variational free energy, and metrics appear in geometric formulations of the free energy principle. The standard algorithms use ordinary gradients, so the metric view is an extension and a reinterpretation.

## Related concepts
- [[riemannian_metric]] — the same structure treated as a Riemannian metric
- [[inner_products]] — the pointwise structure of which the metric tensor is a field
- [[geodesic_equations]] — curves of extremal length determined by the metric
- [[fisher_rao_metric]] — the metric on spaces of probability distributions
- [[tensor_calculus]] — the calculus of such objects under coordinate change

## References
- Lee, J. M. (2018). Introduction to Riemannian Manifolds (2nd ed.). Springer.
- do Carmo, M. P. (1992). Riemannian Geometry. Birkhauser.
- Amari, S., & Nagaoka, H. (2000). Methods of Information Geometry. American Mathematical Society and Oxford University Press.
