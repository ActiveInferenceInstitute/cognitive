---
title: Fisher-Rao Metric
description: The Riemannian metric on a family of probability distributions given by the Fisher information matrix, its invariance properties, closed forms for Gaussian and categorical families, and links to natural gradients.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_geometry
  - riemannian_geometry
  - statistics
semantic_relations:
  - type: relates
    links:
      - "[[fisher_information]]"
      - "[[fisher_metric]]"
      - "[[information_geometry]]"
      - "[[statistical_manifolds]]"
      - "[[natural_gradients]]"
---

# Fisher-Rao Metric

## Overview
A parametric family $\{p_\theta\}$ of probability distributions can be treated as a manifold with coordinates $\theta$. The Fisher-Rao metric makes this manifold Riemannian by taking the Fisher information matrix as the metric tensor,
$$g_{ij}(\theta)=\mathbb{E}_{p_\theta}\!\left[\partial_i\ln p_\theta\,\partial_j\ln p_\theta\right]=-\,\mathbb{E}_{p_\theta}\!\left[\partial_i\partial_j\ln p_\theta\right],$$
where the second form holds under standard regularity conditions. The idea goes back to Rao (1945), and the metric is also described on the page [[fisher_metric]].

## Properties
- **Local meaning.** For small changes, $D_{\mathrm{KL}}(p_\theta\,\|\,p_{\theta+d\theta})\approx\tfrac12\,d\theta^\top g(\theta)\,d\theta$, so distance counts distinguishability of distributions.
- **Invariance.** The metric does not change under reparametrisation of $\theta$ or under sufficient statistics. Cencov's theorem states that, on families of distributions over a finite set, it is the unique such metric up to a constant factor.
- **Cramer-Rao link.** The inverse $g^{-1}$ lower-bounds the covariance of unbiased estimators.

## Examples
For the univariate Gaussian with parameters $(\mu,\sigma)$ one finds $g=\mathrm{diag}(1/\sigma^2,\,2/\sigma^2)$, so $ds^2=(d\mu^2+2\,d\sigma^2)/\sigma^2$, the line element of a hyperbolic plane up to rescaling. For categorical distributions, the substitution $x_i=2\sqrt{p_i}$ maps the simplex to a portion of a sphere of radius two, giving the distance $d(p,q)=2\arccos\sum_i\sqrt{p_iq_i}$.

## Relation to Active Inference
The link is a developed research programme. Variational beliefs live on a statistical manifold, and the Fisher-Rao metric fixes how to measure change in beliefs independently of parametrisation. Steepest descent on variational free energy with respect to this metric is the natural gradient, and geometric treatments of the free energy principle describe belief updating as flow on this manifold. Standard Active Inference schemes use ordinary gradients or fixed-point updates, so the geometric reading is an interpretation and extension rather than a requirement.

## Related concepts
- [[fisher_information]] — the matrix that defines the metric
- [[information_geometry]] — the broader geometric theory of statistical models
- [[natural_gradients]] — steepest descent under this metric
- [[kl_divergence]] — locally quadratic with this metric as its Hessian
- [[cramer_rao_bound]] — bound on estimator variance in terms of the metric
- [[geodesic_equations]] — the equations for shortest paths under the metric

## References
- Rao, C. R. (1945). Information and the accuracy attainable in the estimation of statistical parameters. Bulletin of the Calcutta Mathematical Society, 37, 81-91.
- Amari, S., & Nagaoka, H. (2000). Methods of Information Geometry. American Mathematical Society and Oxford University Press.
- Amari, S. (2016). Information Geometry and Its Applications. Springer.
