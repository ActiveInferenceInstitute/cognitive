---
title: Cramér-Rao Bound
description: The lower bound on the variance of unbiased estimators set by Fisher information, its multivariate form, when it is attained, and its relevance to precision and information geometry.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - estimation_theory
  - fisher_information
  - statistics
semantic_relations:
  - type: relates
    links:
      - "[[fisher_information]]"
      - "[[information_geometry]]"
      - "[[parameter_estimation]]"
      - "[[maximum_likelihood]]"
      - "[[statistical_foundations]]"
---

# Cramér-Rao Bound

## Overview
The Cramér-Rao bound states that no unbiased estimator of a parameter can have variance below the reciprocal of the Fisher information. It was derived independently by Rao (1945) and Cramér (1946), and it gives a benchmark against which the accuracy of any estimation procedure can be judged.

## Statement
Let $x$ be drawn from $p(x\mid\theta)$ with scalar parameter $\theta$, and let $\hat\theta(x)$ be unbiased, so $\mathbb{E}[\hat\theta]=\theta$. Under regularity conditions that allow differentiation under the integral sign,

$$\mathrm{Var}(\hat\theta)\;\ge\;\frac{1}{I(\theta)},\qquad I(\theta)=\mathbb{E}\!\left[\left(\frac{\partial}{\partial\theta}\ln p(x\mid\theta)\right)^{2}\right].$$

For $n$ independent observations the information adds, so the bound becomes $1/(nI(\theta))$. For a vector parameter, $\mathrm{Cov}(\hat{\boldsymbol\theta})-I(\boldsymbol\theta)^{-1}$ is positive semidefinite, where $I(\boldsymbol\theta)$ is the [[fisher_information]] matrix. For a biased estimator with bias $b(\theta)$ the numerator becomes $(1+b'(\theta))^{2}$.

## Attainment and efficiency
An unbiased estimator that reaches the bound is called efficient. Equality holds exactly when the score function is a linear function of the estimator, which happens for the mean-value parameter of an exponential family. For a Gaussian with known variance $\sigma^2$, the sample mean has variance $\sigma^2/n$ and attains the bound. In general, the maximum likelihood estimator is asymptotically efficient, in that its variance approaches the bound as $n$ grows (see [[maximum_likelihood]]). The bound does not apply to estimators that trade bias for lower variance, and it says nothing about small-sample behaviour when the regularity conditions fail.

## Relation to Active Inference and the vault's themes
The link is conceptual and geometric rather than a derivation of Active Inference. The bound makes precise the idea that the curvature of the log-likelihood determines how well a parameter can be resolved, which parallels the role of precision in Gaussian generative models, where the inverse of a precision matrix is the posterior covariance. Fisher information also serves as a Riemannian metric on the space of distributions, so the bound is a statement in [[information_geometry]]. Literature on the free energy principle uses this metric when describing how beliefs change along a statistical manifold. Claims that the brain operates near the bound are hypotheses about specific systems, not consequences of the theorem.

## Related concepts
- [[fisher_information]] — the quantity that sets the bound
- [[information_geometry]] — Fisher information as a metric on distributions
- [[parameter_estimation]] — the problem the bound constrains
- [[maximum_likelihood]] — the asymptotically efficient estimator
- [[precision_weighting]] — precision as inverse variance in predictive models

## References
- Rao, C. R. (1945). Information and the accuracy attainable in the estimation of statistical parameters. Bulletin of the Calcutta Mathematical Society, 37, 81-91.
- Cramér, H. (1946). Mathematical Methods of Statistics. Princeton University Press.
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
