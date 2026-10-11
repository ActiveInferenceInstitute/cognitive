---
title: Statistical Manifolds
type: mathematical_concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - information_geometry
  - differential_geometry
  - statistics
semantic_relations:
  - type: relates
    links:
      - "[[knowledge_base/mathematics/information_geometry]]"
      - "[[knowledge_base/mathematics/fisher_information]]"
      - "[[knowledge_base/mathematics/natural_gradients]]"
      - "[[knowledge_base/mathematics/exponential_family]]"
aliases:
  - statistical_manifold
---

# Statistical Manifolds

## Overview
A statistical manifold is a family of probability distributions $\{p(x;\theta)\}$ whose parameters $\theta\in\Theta\subseteq\mathbb{R}^d$ are treated as coordinates on a smooth manifold. Each point is a whole distribution, and geometric notions (distance, curvature, geodesics) are defined so that they do not depend on how the family is parameterised. The standard Riemannian metric on such a manifold is the Fisher information metric.

## Definition / Formalism
The Fisher metric has components

$$g_{ij}(\theta)=\mathbb{E}_{p(x;\theta)}\!\left[\partial_i \ln p(x;\theta)\,\partial_j \ln p(x;\theta)\right].$$

It is the local quadratic form of the KL divergence: for a small displacement $\delta$,

$$D_{\mathrm{KL}}\big(p_\theta\,\|\,p_{\theta+\delta}\big)\approx \tfrac12\,\delta^{\top} g(\theta)\,\delta .$$

Worked cases (each checked numerically against the KL expansion):

- Bernoulli$(\theta)$: $g(\theta)=1/[\theta(1-\theta)]$.
- Gaussian in coordinates $(\mu,\sigma)$: $g=\mathrm{diag}(1/\sigma^2,\;2/\sigma^2)$.
- Categorical distributions $p\in\Delta^{n-1}$: the map $p\mapsto 2\sqrt{p}$ sends the simplex isometrically onto part of a sphere of radius 2, so the geodesic (Fisher-Rao) distance is $d(p,q)=2\arccos\sum_i\sqrt{p_iq_i}$. For $p=(0.2,0.5,0.3)$ and $q=(0.4,0.4,0.2)$ this is about 0.448.

Exponential families have a dually flat structure: the natural parameters and the expectation parameters are two affine coordinate systems related by Legendre duality, which is why Bregman-type KL geometry is especially simple there. The steepest-descent direction under this metric is the natural gradient $g^{-1}\nabla_\theta L$.

## Role in Active Inference
The link is real but mostly conceptual. Beliefs in a discrete generative model live on a product of simplices, so belief updating is a motion on a statistical manifold, and variational free energy is a function on it. Natural-gradient updates and the geometric reading of precision and policy selection use this structure. The package in `code/tools/src` does not compute Fisher metrics or geodesics; its updates are direct normalised-exponential and matrix operations on the simplex. Treat the geometric descriptions of the free-energy principle as a theoretical lens, not as something the repo implements.

## Related concepts
- [[information_geometry]] — the broader theory built on statistical manifolds
- [[fisher_information]] — the metric tensor itself
- [[natural_gradients]] — steepest descent measured in that metric
- [[exponential_family]] — the dually flat families
- [[differential_geometry]] — manifold, metric, and geodesic machinery
- [[kl_divergence]] — its second-order expansion gives the metric
- [[optimal_transport]] — an alternative, support-aware geometry on distributions

## References
- Amari, S. and Nagaoka, H. (2000). *Methods of Information Geometry*. American Mathematical Society / Oxford University Press.
- Amari, S. (2016). *Information Geometry and Its Applications*. Springer.
