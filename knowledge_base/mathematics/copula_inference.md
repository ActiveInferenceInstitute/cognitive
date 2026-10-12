---
title: Copula Inference
description: Copula inference models multivariate dependence separately from the marginal distributions, using Sklar's theorem, and covers estimation of copula parameters and use of copulas in variational approximations.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistics
  - dependence_modeling
  - variational_inference
semantic_relations:
  - type: relates
    links:
      - "[[probability_distributions]]"
      - "[[mutual_information]]"
      - "[[mean_field_theory]]"
      - "[[variational_inference]]"
      - "[[probability_theory]]"
---

# Copula Inference

## Overview
A copula is a multivariate distribution function on the unit cube whose one-dimensional margins are uniform. Copulas separate the description of a joint distribution into the marginal distributions of each variable and a function that captures the dependence between them. Copula inference is the set of statistical methods for choosing, estimating and using such dependence models.

## Sklar's theorem
For a joint distribution function $F$ with margins $F_1,\dots,F_d$, there exists a copula $C$ such that
$$F(x_1,\dots,x_d)=C\big(F_1(x_1),\dots,F_d(x_d)\big),$$
and $C$ is unique when the margins are continuous. When $C$ and the margins have densities, the joint density factorises as $f(x)=c(F_1(x_1),\dots,F_d(x_d))\prod_i f_i(x_i)$. Because $C$ depends only on the ranks of the data, it is invariant under strictly increasing transformations of the variables. Rank-based dependence measures such as Kendall's $\tau$ and Spearman's $\rho$ are functions of the copula alone. The Pearson correlation is not, and the mutual information of the variables equals the negative differential entropy of the copula density.

## Families and estimation
Common parametric families are the Gaussian and Student-$t$ copulas, which are elliptical, and Archimedean copulas such as Clayton, Gumbel and Frank, which differ in their tail dependence. Vine copulas build a high-dimensional dependence model from a cascade of bivariate copulas. Estimation by full maximum likelihood fits margins and copula jointly. The inference-functions-for-margins method fits the margins first and then the copula parameters, which is computationally simpler but less efficient. A semiparametric alternative replaces the margins by empirical distribution functions and works with rank-transformed pseudo-observations. Goodness of fit is assessed with tests based on the empirical copula.

## Relation to Active Inference
Mean-field variational approximations factorise the posterior and so discard dependence between latent variables. Copula variational inference restores some of it by placing a copula over the factorised marginals. Standard Active Inference schemes use mean-field or structured factorisations and do not rely on copulas, so the connection is a possible refinement of the approximate posterior, not part of the established formalism.

## Related concepts
- [[variational_inference]] — the setting in which copula posteriors are used
- [[mean_field_theory]] — the factorised approximation that copulas extend
- [[mutual_information]] — dependence measure expressed through the copula density
- [[probability_distributions]] — the margins and joint laws being combined
- [[random_variables]] — the variables whose dependence is modelled
- [[probability_theory]] — the underlying measure-theoretic framework

## References
- Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges. Publications de l'Institut de Statistique de l'Université de Paris, 8, 229-231.
- Nelsen, R. B. (2006). An Introduction to Copulas (2nd ed.). Springer.
- Joe, H. (2014). Dependence Modeling with Copulas. Chapman and Hall/CRC.
- Tran, D., Blei, D., & Airoldi, E. M. (2015). Copula variational inference. Advances in Neural Information Processing Systems, 28.
