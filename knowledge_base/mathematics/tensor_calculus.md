---
title: Tensor Calculus
description: The calculus of tensor fields on manifolds, covering index notation, transformation laws, the metric, covariant differentiation and curvature, as the language of differential geometry.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - differential_geometry
  - tensors
  - index_notation
semantic_relations:
  - type: relates
    links:
      - "[[differential_geometry]]"
      - "[[metric_tensor]]"
      - "[[riemannian_metric]]"
      - "[[geodesic_equations]]"
      - "[[linear_algebra]]"
wikidata: Q4454681
---

# Tensor Calculus

## Overview
Tensor calculus is a systematic way of writing and differentiating geometric quantities so that equations keep the same form in every coordinate system. A tensor of type $(r,s)$ is a multilinear map taking $r$ covectors and $s$ vectors to a number. In coordinates it is represented by components $T^{i_1\dots i_r}{}_{j_1\dots j_s}$, with upper indices for contravariant and lower indices for covariant slots. The Einstein summation convention implies a sum over any index appearing once up and once down.

## Transformation and algebra
Under a change of coordinates $x\mapsto x'$, vector components transform with the Jacobian, $v'^i=\frac{\partial x'^i}{\partial x^k}v^k$, and covector components with its inverse, $w'_j=\frac{\partial x^l}{\partial x'^j}w_l$. A general tensor transforms with one factor of each kind per index, so that an equation between tensors holds in all coordinates once it holds in one. Operations that preserve this property are addition, the tensor product, and contraction of an upper with a lower index. A metric $g_{ij}$ gives an isomorphism between vectors and covectors, with $v_i=g_{ij}v^j$ and $v^i=g^{ij}v_j$ ([[metric_tensor]]).

## Covariant differentiation
Partial derivatives of tensor components do not transform as tensors, because they ignore the variation of the basis. The covariant derivative corrects this with a connection. For the Levi-Civita connection of a metric, the Christoffel symbols are
$$\Gamma^k_{ij}=\tfrac12g^{kl}\big(\partial_ig_{jl}+\partial_jg_{il}-\partial_lg_{ij}\big),$$
and for a vector field
$$\nabla_iv^j=\partial_iv^j+\Gamma^j_{ik}v^k.$$
The Levi-Civita connection is the unique torsion-free connection for which $\nabla g=0$. Geodesics are curves with $\nabla_{\dot\gamma}\dot\gamma=0$ ([[geodesic_equations]]). Failure of covariant derivatives to commute defines the Riemann curvature tensor, from which the Ricci tensor and scalar curvature are obtained by contraction (Lee, 2018).

## Relation to Active Inference
Tensor notation is used in information geometry, where the Fisher information is a metric tensor on the manifold of probability distributions and the natural gradient is defined by $\tilde\nabla f=G^{-1}\nabla f$, converting the covector $df$ into a vector using the inverse metric (Amari, 1998). Geometric treatments of belief dynamics rely on this language ([[riemannian_metric]], [[natural_gradients]], [[geometric_active_inference]]). Tensors in the sense of multi-dimensional arrays, as used for likelihood arrays in discrete models, are a related but simpler usage: they lack the transformation law that defines geometric tensors.

## Related concepts
- [[differential_geometry]] — the field that uses tensor calculus as its language
- [[metric_tensor]] — the tensor that gives geometry its structure
- [[riemannian_metric]] — a smoothly varying metric on a manifold
- [[geodesic_equations]] — curves defined by the connection
- [[natural_gradients]] — gradient converted by an inverse metric
- [[linear_algebra]] — the algebraic foundation for tensors

## References
- Lee, J. M. (2018). Introduction to Riemannian Manifolds (2nd ed.). Springer.
- Schutz, B. F. (1980). Geometrical Methods of Mathematical Physics. Cambridge University Press.
- Amari, S. (1998). Natural gradient works efficiently in learning. Neural Computation, 10, 251-276.
