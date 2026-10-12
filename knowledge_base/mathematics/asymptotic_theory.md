---
title: Asymptotic Theory
description: The study of statistical procedures as the sample size grows, covering consistency, asymptotic normality of maximum likelihood estimators, likelihood ratio tests, the Bernstein-von Mises theorem and Laplace approximations.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistics
  - estimation
  - large_sample_theory
semantic_relations:
  - type: relates
    links:
      - "[[maximum_likelihood]]"
      - "[[fisher_information]]"
      - "[[cramer_rao_bound]]"
      - "[[likelihood_function]]"
      - "[[laplace_approximation]]"
      - "[[bayesian_statistics]]"
---

# Asymptotic Theory

## Overview
Asymptotic theory describes the behaviour of estimators and tests as the number of observations $n$ tends to infinity. Exact finite-sample distributions are often intractable, whereas limiting distributions are simple and give approximate confidence intervals, tests and error bounds. The results rest on regularity conditions, which typically include a correctly specified, identifiable and smooth model with parameters in the interior of the parameter space.

## Main results
Let $X_1,\dots,X_n$ be independent and identically distributed with density $p(x;\theta_0)$, and let $\hat\theta_n$ be the maximum likelihood estimator (MLE).
- **Consistency**: $\hat\theta_n\to\theta_0$ in probability.
- **Asymptotic normality**: 
$$\sqrt n\,(\hat\theta_n-\theta_0)\xrightarrow{d}\mathcal N\big(0,\,I(\theta_0)^{-1}\big),$$
where $I(\theta)=\mathbb E\big[\nabla\log p\,\nabla\log p^{\top}\big]$ is the Fisher information of one observation (see [[fisher_information]]).
- **Efficiency**: the limiting covariance $I(\theta_0)^{-1}$ equals the Cramér-Rao lower bound for a single observation, so no regular estimator has smaller asymptotic variance (see [[cramer_rao_bound]]).
- **Likelihood ratio test**: for a null hypothesis fixing $k$ parameters, Wilks' theorem gives $2\{\ell(\hat\theta)-\ell(\hat\theta_0)\}\xrightarrow{d}\chi^2_k$ under the null, where $\ell$ is the log-likelihood and $\hat\theta_0$ the restricted estimate.
- **Delta method**: if $\sqrt n(\hat\theta_n-\theta_0)\to\mathcal N(0,\Sigma)$ and $g$ is differentiable, then $\sqrt n\big(g(\hat\theta_n)-g(\theta_0)\big)\to\mathcal N\big(0,\nabla g^{\top}\Sigma\nabla g\big)$.
- **Bernstein-von Mises theorem**: under regularity and a prior with positive density near $\theta_0$, the posterior is asymptotically normal, centred near the MLE with covariance about $(nI(\theta_0))^{-1}$, and the effect of the prior vanishes. Credible regions then agree asymptotically with confidence regions.

Failures occur outside the conditions, for example when the model is misspecified, when parameters lie on a boundary, when the dimension grows with $n$, or when the model is singular, as in mixtures and neural networks, where the likelihood is not locally quadratic.

## Relation to Active Inference
Active Inference implementations often use the Laplace approximation, which assumes a Gaussian posterior around a mode. The Bernstein-von Mises theorem is the asymptotic justification of that assumption for regular models with ample data, and a related expansion of the log evidence underlies criteria such as the Bayesian information criterion (see [[laplace_approximation]]). The link is a mathematical justification of approximation methods used in the framework, not a property of the free energy principle. For small samples, multimodal posteriors or hierarchical models with singularities, the Gaussian approximation can be poor.

## Related concepts
- [[maximum_likelihood]] — the estimator whose limit theory is central
- [[fisher_information]] — determines the asymptotic variance
- [[cramer_rao_bound]] — the variance bound the MLE attains asymptotically
- [[likelihood_function]] — the object on which these results are built
- [[laplace_approximation]] — Gaussian approximation justified asymptotically
- [[bayesian_statistics]] — posterior consistency and normality

## References
- van der Vaart, A. W. (1998). Asymptotic statistics. Cambridge University Press.
- Lehmann, E. L., & Casella, G. (1998). Theory of point estimation (2nd ed.). Springer.
- Casella, G., & Berger, R. L. (2002). Statistical inference (2nd ed.). Duxbury.
