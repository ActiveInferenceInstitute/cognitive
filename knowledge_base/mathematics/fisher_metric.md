---
title: Fisher Metric
description: The Riemannian metric on a space of probability distributions given by the Fisher information matrix, its invariance properties, relation to KL divergence, and use in natural gradients and information geometry.
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
      - "[[information_geometry]]"
      - "[[statistical_manifolds]]"
      - "[[natural_gradients]]"
      - "[[kl_divergence]]"
---

# Fisher Metric

## Overview
For a parametric family $p(x;\theta)$, the Fisher metric defines an inner product on the tangent space at each parameter value, with components given by the Fisher information matrix
$$
g_{ij}(\theta)=\mathbb{E}_{x\sim p(\cdot;\theta)}\Big[\partial_i\ln p(x;\theta)\,\partial_j\ln p(x;\theta)\Big].
$$
This turns the family into a Riemannian manifold, called a statistical manifold, in which distance between nearby distributions reflects how distinguishable they are from data. The use of the Fisher information as a geometric quantity goes back to Rao (1945). The metric is positive semi-definite, and positive definite when the parameterisation is identifiable.

## Properties
- **Local form of KL divergence**: for a small change $d\theta$, $D_{\mathrm{KL}}\big(p_\theta\,\|\,p_{\theta+d\theta}\big)\approx\tfrac12\,d\theta^{\top}g(\theta)\,d\theta$, so the metric measures divergence to second order.
- **Reparameterisation covariance**: under a smooth change of parameters it transforms as a rank-two covariant tensor, so lengths and angles do not depend on the coordinates chosen.
- **Uniqueness**: by Chentsov's theorem, on finite sample spaces it is, up to a constant factor, the only Riemannian metric that is invariant under sufficient statistics (Amari & Nagaoka, 2000).
- **Example**: for a univariate Gaussian with parameters $(\mu,\sigma)$, $g=\mathrm{diag}(1/\sigma^2,\,2/\sigma^2)$, which is a hyperbolic geometry up to scaling.

## Natural gradient
Steepest descent with respect to the Fisher metric, $\Delta\theta\propto -g(\theta)^{-1}\nabla_\theta\mathcal{L}$, is the natural gradient (Amari, 1998). It does not depend on the parameterisation and often converges faster than the ordinary gradient on ill-conditioned problems. The cost of inverting $g$ motivates structured approximations in large models.

## Relation to Active Inference
The Fisher metric supplies the geometry in which belief updating can be described. Variational free energy minimisation with respect to the parameters of an approximate posterior can be written as a natural-gradient flow, and differential-geometric accounts of the free energy principle describe the dynamics of internal states as motion on the statistical manifold of beliefs. Parr, Da Costa and Friston (2020) develop this link between Markov blankets, information geometry and stochastic thermodynamics. These are mathematical reformulations that clarify structure; they are an active research direction, not an empirically separate prediction.

## Related concepts
- [[fisher_information]] — the matrix that defines the metric
- [[information_geometry]] — the broader field built on it
- [[statistical_manifolds]] — the spaces on which it acts
- [[natural_gradients]] — optimisation steps measured in this metric
- [[cramer_rao_bound]] — the estimation bound set by its inverse

## References
- Rao, C. R. (1945). Information and the accuracy attainable in the estimation of statistical parameters. Bulletin of the Calcutta Mathematical Society, 37, 81-91.
- Amari, S. (1998). Natural gradient works efficiently in learning. Neural Computation, 10, 251-276.
- Amari, S., & Nagaoka, H. (2000). Methods of Information Geometry. American Mathematical Society and Oxford University Press.
- Parr, T., Da Costa, L., & Friston, K. (2020). Markov blankets, information geometry and stochastic thermodynamics. Philosophical Transactions of the Royal Society A, 378, 20190159.
