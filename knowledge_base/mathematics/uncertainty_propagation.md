---
title: Uncertainty Propagation
description: How uncertainty in the inputs of a function or dynamical model is transferred to its outputs, covering first-order linearisation, sampling-based and unscented methods, and the Gaussian case used in filtering.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - uncertainty
  - error_analysis
  - probability
semantic_relations:
  - type: relates
    links:
      - "[[uncertainty_quantification]]"
      - "[[error_propagation]]"
      - "[[monte_carlo_methods]]"
      - "[[probability_distributions]]"
      - "[[variational_free_energy]]"
---

# Uncertainty Propagation

## Overview
Uncertainty propagation asks how the distribution of an output $y=f(\mathbf{x})$ follows from the distribution of an uncertain input $\mathbf{x}$. It is one part of [[uncertainty_quantification]], alongside characterising the input uncertainty and attributing output variation to its sources. In dynamical models, errors accumulate over time, so small initial uncertainty can grow, as in chaotic systems, or shrink, as in stable ones.

## Methods
- **First-order (delta method).** For $\mathbf{x}$ with mean $\boldsymbol{\mu}$ and covariance $\Sigma$, linearise $f$ about $\boldsymbol{\mu}$ with Jacobian $J$:
$$\mathbb{E}[\mathbf{y}]\approx f(\boldsymbol{\mu}),\qquad \mathrm{Cov}(\mathbf{y})\approx J\,\Sigma\,J^{\top}.$$
For a scalar function of independent variables this gives $\sigma_y^2\approx\sum_i(\partial f/\partial x_i)^2\sigma_i^2$, the standard rule used in laboratory error analysis. The approximation is exact for affine $f$ with Gaussian input and degrades with strong nonlinearity or large variance, where a second-order correction to the mean, $\tfrac12\mathrm{tr}(H\Sigma)$ with Hessian $H$, may be needed.
- **Monte Carlo.** Draw samples of $\mathbf{x}$, evaluate $f$ on each, and summarise the outputs. It makes no linearity assumption, and its error falls as $N^{-1/2}$ independently of dimension, but it requires many model evaluations (see [[monte_carlo_methods]]).
- **Unscented transform.** A deterministic set of sigma points chosen to match the input mean and covariance is passed through $f$, and the output moments are estimated from them. It is accurate to higher order than linearisation for the same cost scale.
- **Polynomial chaos.** The output is expanded in orthogonal polynomials of the input random variables, so that moments follow from the coefficients.

## Dynamics
For a linear system $\mathbf{x}_{t+1}=F\mathbf{x}_t+\mathbf{w}_t$ with noise covariance $Q$, the covariance evolves as $\Sigma_{t+1}=F\Sigma_tF^{\top}+Q$, which is the prediction step of the Kalman filter. Nonlinear filters apply the methods above at each step.

## Relation to Active Inference
Active Inference models represent beliefs with an explicit uncertainty, usually as a mean and precision under a Gaussian (Laplace) approximation, and these are propagated through the generative model's nonlinear mappings in the same first-order manner. The precision weighting of prediction errors is the mechanism that decides how much input uncertainty is passed on to belief updates. The relation is one of shared mathematics in standard implementations, and it is approximate when the model is strongly nonlinear.

## Related concepts
- [[uncertainty_quantification]] — the wider programme of which propagation is a part
- [[error_propagation]] — the classical, first-order treatment of measurement errors
- [[monte_carlo_methods]] — sampling-based propagation
- [[probability_distributions]] — the objects whose transformation is being tracked
- [[variational_free_energy]] — an objective whose Laplace form uses propagated covariances

## References
- Joint Committee for Guides in Metrology (2008). Evaluation of measurement data: Guide to the expression of uncertainty in measurement (JCGM 100:2008).
- Bevington, P. R., & Robinson, D. K. (2003). Data Reduction and Error Analysis for the Physical Sciences (3rd ed.). McGraw-Hill.
- Julier, S. J., & Uhlmann, J. K. (2004). Unscented filtering and nonlinear estimation. Proceedings of the IEEE, 92, 401-422.
- Sullivan, T. J. (2015). Introduction to Uncertainty Quantification. Springer.
