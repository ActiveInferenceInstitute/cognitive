---
title: Riemannian Metric
description: A smoothly varying inner product on the tangent spaces of a manifold, used to define lengths, angles and geodesics, with the Fisher information metric as the principal example for statistical models.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - differential_geometry
  - information_geometry
  - metric
semantic_relations:
  - type: relates
    links:
      - "[[fisher_information]]"
      - "[[metric_tensor]]"
      - "[[differential_geometry]]"
      - "[[geodesics]]"
      - "[[statistical_manifolds]]"
wikidata: Q7888899
---

# Riemannian Metric

## Overview
A Riemannian metric on a smooth manifold $M$ assigns to each point $p$ an inner product $g_p$ on the tangent space $T_pM$, in such a way that $g_p$ varies smoothly with $p$. Each $g_p$ is symmetric, bilinear and positive definite. A manifold together with such a metric is a Riemannian manifold. The metric turns purely topological and differentiable structure into geometry: it supplies lengths of tangent vectors, angles between them, and distances between points.

## Definition and consequences
In local coordinates $(x^1,\dots,x^n)$ the metric is a matrix of smooth functions $g_{ij}(x)$, and the squared length of a tangent vector $v$ is $g_{ij}v^iv^j$ with summation over repeated indices ([[metric_tensor]], [[tensor_calculus]]). Several constructions follow.
- **Length and distance.** The length of a curve $\gamma:[a,b]\to M$ is $L(\gamma)=\int_a^b\sqrt{g(\dot\gamma,\dot\gamma)}\,dt$. The distance between two points is the infimum of the lengths of curves joining them, which makes a connected Riemannian manifold a metric space.
- **Geodesics.** Locally length-minimising curves satisfy $\ddot x^k+\Gamma^k_{ij}\dot x^i\dot x^j=0$, with Christoffel symbols $\Gamma^k_{ij}=\tfrac12g^{kl}(\partial_ig_{jl}+\partial_jg_{il}-\partial_lg_{ij})$ ([[geodesic_equations]]).
- **Volume and gradient.** The metric defines the volume element $\sqrt{\det g}\,dx$ and converts the differential $df$ into the gradient vector field with components $g^{ij}\partial_jf$. The choice of metric therefore changes which direction counts as steepest descent.
- **Existence.** Every smooth manifold admits a Riemannian metric, and the metric is not unique. Different choices give different geometries on the same manifold.

## The Fisher metric
For a parametric family $p(x\mid\theta)$, the Fisher information matrix
$$g_{ij}(\theta)=\mathbb{E}_\theta\big[\partial_i\ln p(x\mid\theta)\,\partial_j\ln p(x\mid\theta)\big]$$
defines a Riemannian metric on parameter space. Chentsov's theorem states that, for models on finite sample spaces, it is the unique metric up to a constant factor that is invariant under sufficient statistics, which motivates its use in [[information_geometry]] (Amari & Nagaoka, 2000). See [[fisher_information]] and [[fisher_metric]].

## Relation to Active Inference
The link is a well-developed formal one. Free energy minimisation is often described as gradient flow on a statistical manifold, and replacing the Euclidean gradient with the gradient under the Fisher metric yields the natural gradient ([[natural_gradients]]). Treatments of belief updating as motion on the manifold of beliefs use this geometry, see [[geometric_active_inference]]. The geometric picture is a mathematical description of the dynamics, and does not by itself establish that neural implementations compute natural gradients.

## Related concepts
- [[fisher_information]] — the source of the Fisher metric
- [[metric_tensor]] — the coordinate representation of the metric
- [[geodesics]] — shortest paths under the metric
- [[statistical_manifolds]] — manifolds of probability distributions
- [[natural_gradients]] — steepest descent under a Riemannian metric
- [[differential_geometry]] — the surrounding theory

## References
- Lee, J. M. (2018). Introduction to Riemannian Manifolds (2nd ed.). Springer.
- Amari, S., & Nagaoka, H. (2000). Methods of Information Geometry. American Mathematical Society and Oxford University Press.
