---
title: Gaussian Distributions
description: The normal distribution and its multivariate form, covering density, closure under linear operations, maximum entropy, and the central role of Gaussian assumptions in Bayesian inference and Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - probability_distributions
  - statistics
  - bayesian_inference
semantic_relations:
  - type: relates
    links:
      - "[[probability_distributions]]"
      - "[[exponential_family]]"
      - "[[laplace_approximation]]"
      - "[[conjugate_priors]]"
      - "[[maximum_entropy_principle]]"
---

# Gaussian Distributions

## Overview
The Gaussian or normal distribution is the continuous distribution fully described by a mean and a covariance. Its prevalence comes from the central limit theorem, from its closure under common operations and from its analytical tractability. For a $k$-dimensional random vector with mean $\mu$ and positive definite covariance $\Sigma$ the density is
$$\mathcal{N}(x;\mu,\Sigma) = (2\pi)^{-k/2}\,|\Sigma|^{-1/2}\exp\!\left(-\tfrac12 (x-\mu)^\top \Sigma^{-1}(x-\mu)\right).$$

## Key properties
- **Linear closure**: if $x$ is Gaussian then $Ax+b$ is Gaussian with mean $A\mu+b$ and covariance $A\Sigma A^\top$.
- **Marginals and conditionals**: partition $x=(a,b)$. The marginal of $a$ is Gaussian, and the conditional has mean $\mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(b-\mu_b)$ and covariance $\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}$.
- **Maximum entropy**: among distributions on $\mathbb{R}^k$ with given mean and covariance, the Gaussian has the largest differential entropy, $\tfrac12\ln\big((2\pi e)^k|\Sigma|\big)$.
- **Exponential family**: it is a member of the exponential family, with sufficient statistics $x$ and $xx^\top$.
- **Conjugacy**: a Gaussian likelihood with known variance has a Gaussian conjugate prior for the mean.
- **Zero correlation and independence**: for jointly Gaussian variables these coincide.

## Role in Active Inference
Gaussian assumptions appear at several points. In continuous-state generative models, observation and state noise are usually taken to be Gaussian, which makes the log density quadratic and prediction errors precision-weighted sums of squares. Under the Laplace approximation the variational density is a Gaussian centred on a mode, so that belief updating reduces to gradient descent on a quadratic form with curvature setting the covariance. Precision is the inverse of the variance, which is where precision weighting enters. These choices are modelling assumptions that buy tractability, not claims that the world is Gaussian. Discrete-state models in the POMDP formulation use categorical distributions instead.

## Related concepts
- [[probability_distributions]] — the wider family of distributions
- [[laplace_approximation]] — approximates a posterior by a Gaussian at its mode
- [[exponential_family]] — the class containing Gaussians
- [[gaussian_processes]] — distributions over functions with Gaussian marginals
- [[precision_weighting]] — scaling of errors by inverse variance
- [[kl_divergence]] — has closed form between two Gaussians

## References
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
- MacKay, D. J. C. (2003). Information Theory, Inference, and Learning Algorithms. Cambridge University Press.
