---
title: Natural Gradient Flow
description: Steepest-descent dynamics on a statistical manifold with respect to the Fisher information metric, giving parametrisation-invariant optimisation and a geometric reading of variational updates.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_geometry
  - optimization
  - gradient_flows
semantic_relations:
  - type: relates
    links:
      - "[[natural_gradients]]"
      - "[[fisher_information]]"
      - "[[information_geometry]]"
      - "[[gradient_flows]]"
      - "[[geodesic_flows]]"
      - "[[geometric_active_inference]]"
---

# Natural Gradient Flow

## Overview
Ordinary gradient descent depends on how a model is parametrised: reparametrising $\theta$ changes the direction of steepest descent, because the Euclidean metric on parameters has no statistical meaning. Natural gradient flow replaces that metric with the Fisher information metric of the model family, so that distance between parameter values reflects how distinguishable the corresponding distributions are. The flow is
$$\dot\theta=-G(\theta)^{-1}\nabla_\theta L(\theta),\qquad G_{ij}(\theta)=\mathbb{E}_{p_\theta}\!\big[\partial_i\ln p_\theta\,\partial_j\ln p_\theta\big],$$
where $L$ is a loss or free energy. Its discretisation $\theta\leftarrow\theta-\eta\,G^{-1}\nabla_\theta L$ is the natural gradient method, introduced for learning by Amari (1998).

## Geometric meaning
The Fisher matrix gives the local quadratic form of the Kullback-Leibler divergence: $\mathrm{KL}(p_\theta\|p_{\theta+d\theta})\approx\tfrac12\,d\theta^{\top}G\,d\theta$. The direction $-G^{-1}\nabla L$ is the steepest descent direction when step size is measured by this divergence, which is a Riemannian gradient on the statistical manifold. Because the metric transforms as a tensor, the flow is invariant under smooth reparametrisation of the model, as a gradient flow on a Riemannian manifold should be. For exponential families there is a convenient identity: the natural gradient with respect to the natural parameters equals the ordinary gradient with respect to the expectation parameters, since $\partial\eta/\partial\theta=G$.

A practical obstacle is the cost of forming and inverting $G$, which grows quadratically in the number of parameters for storage and cubically for exact inversion. Approximations (diagonal, block-diagonal or Kronecker-factored estimates) are used in large models.

## Role in Active Inference
Variational inference optimises a free energy over the parameters of an approximate posterior. For exponential-family approximations, a natural gradient step has a simple form, which is the basis of stochastic variational inference (Hoffman et al., 2013), and several message-passing updates can be read as natural gradient steps on the free energy. Geometric accounts of Active Inference describe belief updating as a flow on a statistical manifold with the Fisher metric. The correspondence for specific update schemes is established; the claim that perception in general follows a natural gradient flow is a modelling hypothesis.

## Related concepts
- [[natural_gradients]] — the optimisation method in discrete steps
- [[fisher_information]] — the metric defining the flow
- [[information_geometry]] — geometry of families of distributions
- [[gradient_flows]] — flows on metric spaces and manifolds
- [[geodesic_flows]] — curves of locally shortest distance in the same geometry
- [[variational_inference]] — setting where the free energy is minimised

## References
- Amari, S. (1998). Natural gradient works efficiently in learning. Neural Computation, 10, 251-276.
- Amari, S. (2016). Information Geometry and Its Applications. Springer.
- Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). Stochastic variational inference. Journal of Machine Learning Research, 14, 1303-1347.
