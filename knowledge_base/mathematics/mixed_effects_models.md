---
title: Mixed Effects Models
description: Regression models combining fixed effects shared across a population with random effects that vary by group or subject, covering the Laird-Ware form, partial pooling, estimation by REML and generalised extensions.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistics
  - regression
  - hierarchical_models
semantic_relations:
  - type: relates
    links:
      - "[[hierarchical_models]]"
      - "[[multilevel_modeling]]"
      - "[[hierarchical_bayes]]"
      - "[[probabilistic_models]]"
      - "[[statistical_modeling]]"
---

# Mixed Effects Models

## Overview
Mixed effects models describe data in which observations are grouped, for example repeated measurements on the same participant or students within schools. They combine fixed effects, parameters that apply to the whole population, with random effects, group-specific deviations treated as draws from a distribution. This accounts for within-group correlation that would invalidate ordinary regression and lets groups with little data borrow strength from the rest.

## The linear model
In the form of Laird and Ware (1982), for group $i$ with response vector $y_i$,
$$y_i=X_i\beta+Z_ib_i+\varepsilon_i,\qquad b_i\sim\mathcal{N}(0,D),\quad \varepsilon_i\sim\mathcal{N}(0,\sigma^2I),$$
with $b_i$ and $\varepsilon_i$ independent. Here $\beta$ holds the fixed effects, $X_i$ and $Z_i$ are design matrices and $D$ is the covariance of the random effects. Integrating out $b_i$ gives the marginal distribution
$$y_i\sim\mathcal{N}\bigl(X_i\beta,\ Z_iDZ_i^\top+\sigma^2I\bigr),$$
whose covariance structure captures the within-group dependence.

## Estimation and interpretation
The variance components are estimated by maximum likelihood or by restricted maximum likelihood (REML), which corrects the downward bias of maximum likelihood variance estimates. The predicted random effects, best linear unbiased predictors, are shrunk from the group's raw deviation towards zero, by more when the group has few observations or the between-group variance is small. This partial pooling lies between fitting a separate model to each group and fitting a single pooled model. Generalised linear mixed models replace the Gaussian response by an exponential-family response with a link function, and inference then requires approximations such as the Laplace approximation or numerical quadrature. Software such as the R package lme4 implements these fits (Bates et al., 2015).

## Relation to Active Inference
The link is a shared hierarchical structure. Mixed effects models are the frequentist counterpart of hierarchical Bayesian models, in which group parameters have a population-level prior, and the empirical priors of hierarchical generative models play the same part. In practice, Active Inference studies often analyse subject-level fitted parameters with group-level models of this type, for instance in parametric empirical Bayes analyses. The correspondence is one of mathematical form, not of theory.

## Related concepts
- [[multilevel_modeling]] — closely overlapping terminology emphasising nested structure
- [[hierarchical_models]] — the Bayesian view of group-level distributions
- [[hierarchical_bayes]] — full Bayesian treatment with priors on variance components
- [[empirical_prior]] — the learned population-level prior in hierarchical inference
- [[probabilistic_models]] — the wider family of models of data-generating processes

## References
- Laird, N. M., & Ware, J. H. (1982). Random-effects models for longitudinal data. Biometrics, 38, 963-974.
- Pinheiro, J. C., & Bates, D. M. (2000). Mixed-Effects Models in S and S-PLUS. Springer.
- Bates, D., Maechler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4. Journal of Statistical Software, 67(1), 1-48.
