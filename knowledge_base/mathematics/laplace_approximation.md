---
title: Laplace Approximation
description: Gaussian approximation of a posterior distribution centred on its mode, its covariance and evidence formulas, its limits, and its role as the Laplace assumption in continuous-time Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - approximate_inference
  - gaussian
  - bayesian_inference
semantic_relations:
  - type: relates
    links:
      - "[[bayesian_inference]]"
      - "[[gaussian_distributions]]"
      - "[[variational_inference]]"
      - "[[model_evidence]]"
      - "[[predictive_coding]]"
---

# Laplace Approximation

## Overview
The Laplace approximation replaces an intractable posterior by a Gaussian density centred at the posterior mode. It uses only local information, the location and curvature of the peak of the log posterior, and so requires an optimisation and a second-derivative calculation rather than any integration or sampling.

## Construction
Let $\theta\in\mathbb{R}^d$ be the parameters and $D$ the data. Write $\ell(\theta)=\ln p(D,\theta)$, which equals the log posterior up to an additive constant. Let $\theta^*$ be a maximum of $\ell$ and let
$$H=-\nabla^2\ell(\theta)\big|_{\theta^*}$$
be the negative Hessian there. A second-order Taylor expansion of $\ell$ around $\theta^*$ gives
$$q(\theta)=\mathcal{N}\big(\theta;\,\theta^*,\,H^{-1}\big).$$
Integrating the same quadratic expansion gives an approximation to the log model evidence:
$$\ln p(D)\approx \ln p(D,\theta^*)+\tfrac{d}{2}\ln 2\pi-\tfrac12\ln\det H.$$
The last two terms act as a penalty for model complexity, and the expression is the basis of the Bayesian information criterion in the large-sample limit.

## Limitations
The approximation is accurate when the posterior is unimodal and close to Gaussian, which usually holds for large samples because posteriors concentrate around the mode. It fails for multimodal, strongly skewed or bounded posteriors, and it depends on the parameterisation, since the mode and curvature change under nonlinear reparameterisation. It cannot represent heavy tails.

## Relation to Active Inference
In continuous-time Active Inference and predictive coding, the Laplace assumption is a standing modelling choice: the approximate posterior is taken to be Gaussian, so that free energy becomes a function of the posterior means and precisions alone, and belief updating reduces to gradient descent on prediction errors weighted by precision (Friston et al., 2007). This is a derivation under an explicit assumption, not a claim that neuronal beliefs are Gaussian. Discrete-state formulations use categorical beliefs and do not rely on it.

## Related concepts
- [[variational_inference]] — the Laplace case is a Gaussian variational family with a mode-based fit
- [[gaussian_distributions]] — the family used for the approximation
- [[model_evidence]] — the quantity the approximation estimates
- [[predictive_coding]] — a scheme that adopts the Laplace assumption
- [[bayesian_brain]] — the hypothesis that motivates Gaussian belief codes

## References
- Friston, K., Mattout, J., Trujillo-Barreto, N., Ashburner, J., & Penny, W. (2007). Variational free energy and the Laplace approximation. NeuroImage, 34, 220-234.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning, section 4.4. Springer.
- MacKay, D. J. C. (2003). Information Theory, Inference, and Learning Algorithms. Cambridge University Press.
