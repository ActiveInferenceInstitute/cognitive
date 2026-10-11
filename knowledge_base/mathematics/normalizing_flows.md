---
title: Normalizing Flows
description: Generative and variational models built from invertible transformations with tractable Jacobians, giving exact likelihoods, and their use for flexible posterior approximations in variational inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - generative_models
  - variational_inference
  - density_estimation
semantic_relations:
  - type: relates
    links:
      - "[[variational_inference]]"
      - "[[variational_autoencoders]]"
      - "[[generative_models]]"
      - "[[neural_odes]]"
      - "[[diffusion_models]]"
---

# Normalizing Flows

## Overview
A normalizing flow represents a complicated probability distribution as an invertible, differentiable transformation of a simple one. A latent variable $z$ with a tractable base density $p_Z$, usually a standard Gaussian, is mapped to data by $x=g_\theta(z)$. Because $g_\theta$ is invertible, the density of $x$ is known exactly, which distinguishes flows from [[variational_autoencoders]], where the likelihood is only bounded.

## Change of variables
For an invertible map the density follows from the change-of-variables formula,
$$\log p_X(x)=\log p_Z\big(g_\theta^{-1}(x)\big)+\log\left|\det \frac{\partial g_\theta^{-1}}{\partial x}\right|.$$
A flow is a composition $g=g_K\circ\dots\circ g_1$, so the log-determinants simply add. The design problem is to make each layer expressive while keeping the determinant cheap and the inverse available. Common choices are:
- **Coupling layers**, as in NICE and RealNVP, which transform half of the variables conditioned on the other half and give triangular Jacobians.
- **Autoregressive flows**, where each variable is transformed conditioned on earlier ones, again with triangular Jacobians.
- **Planar and radial flows**, introduced for variational inference (Rezende and Mohamed, 2015).
- **Continuous flows**, defined by an ODE as in [[neural_odes]].

Constraints include that the latent and data dimensions must match and that invertibility limits the architectures available. Training maximises the exact log-likelihood on data, or in the variational setting minimises a divergence to a target.

## Relation to Active Inference
The link is as a tool for the approximate posterior. Variational inference minimises free energy over a family of approximating densities $q(s)$, and a mean-field or single Gaussian family may be too restrictive. Applying a flow to a simple base $q$ enlarges the family while keeping the entropy term computable, which reduces the gap between free energy and the true negative log-evidence (see [[variational_inference]]). Deep implementations of Active Inference can use flows in this role, though standard discrete and Gaussian formulations do not. The correspondence is that of a technique serving the same objective, not a shared theory.

## Related concepts
- [[variational_inference]] — the setting where flexible posteriors are needed
- [[variational_autoencoders]] — latent-variable models that may use flow posteriors
- [[generative_models]] — the broader class of models for data
- [[neural_odes]] — continuous-time flows
- [[diffusion_models]] — an alternative generative approach based on noising and denoising

## References
- Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows. Proceedings of the 32nd International Conference on Machine Learning, 1530-1538.
- Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2021). Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22(57), 1-64.
