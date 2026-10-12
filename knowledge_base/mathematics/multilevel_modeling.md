---
title: Multilevel Modeling
description: Regression for data with nested levels such as individuals within groups, in which parameters at each level have distributions governed by higher levels, producing partial pooling and shrinkage.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistics
  - hierarchical_models
  - bayesian_statistics
semantic_relations:
  - type: relates
    links:
      - "[[mixed_effects_models]]"
      - "[[hierarchical_models]]"
      - "[[hierarchical_bayes]]"
      - "[[hyperprior]]"
      - "[[bayesian_statistics]]"
---

# Multilevel Modeling

## Overview
Multilevel modelling, also called hierarchical linear modelling, treats data that are organised in levels: measurements within individuals, individuals within classrooms, classrooms within schools. Parameters at one level, such as an intercept per classroom, are themselves modelled as draws from a distribution whose parameters are estimated from the data. The approach is closely related to mixed effects models, and the two terms are used for largely overlapping methods. The multilevel label emphasises nested structure and, particularly in the Bayesian literature, the hierarchy of priors.

## A varying-intercept model
For observation $i$ in group $j[i]$,
$$y_i=\alpha_{j[i]}+\beta x_i+\varepsilon_i,\quad \varepsilon_i\sim\mathcal{N}(0,\sigma_y^2),\qquad \alpha_j\sim\mathcal{N}(\mu_\alpha,\sigma_\alpha^2).$$
The group intercepts share a population distribution. For a group with $n_j$ observations and group mean $\bar y_j$ (taking $\beta=0$ for clarity), the estimate is a precision-weighted average
$$\hat\alpha_j=\frac{\dfrac{n_j}{\sigma_y^2}\,\bar y_j+\dfrac{1}{\sigma_\alpha^2}\,\mu_\alpha}{\dfrac{n_j}{\sigma_y^2}+\dfrac{1}{\sigma_\alpha^2}},$$
so groups with few observations are pulled toward the overall mean. This is partial pooling, between the extremes of no pooling, one estimate per group, and complete pooling, ignoring groups. The intraclass correlation $\sigma_\alpha^2/(\sigma_\alpha^2+\sigma_y^2)$ measures the share of variance at the group level.

## Extensions
Slopes can vary by group, with a joint distribution on intercepts and slopes, and predictors can be measured at the group level. Non-nested (crossed) structures, generalised responses and more than two levels are handled in the same framework. In the Bayesian version, hyperpriors are placed on the variance parameters, and the model can be fitted by Markov chain Monte Carlo (Gelman and Hill, 2007).

## Relation to Active Inference
The correspondence is close in mathematical form. Hierarchical generative models in Active Inference have the same architecture: parameters or states at one level act as priors for the level below, and the precision-weighted combination of prior and evidence in the formula above is the same computation as precision-weighted prediction-error updating. The analogy is of form and not a claim that the brain does multilevel regression.

## Related concepts
- [[mixed_effects_models]] — the frequentist formulation of the same structure
- [[hierarchical_models]] — models with parameters organised in levels
- [[hierarchical_active_inference]] — nested generative models in Active Inference
- [[hyperprior]] — prior placed on the parameters of another prior
- [[empirical_prior]] — priors learned from data at a higher level

## References
- Gelman, A., & Hill, J. (2007). Data Analysis Using Regression and Multilevel/Hierarchical Models. Cambridge University Press.
- Raudenbush, S. W., & Bryk, A. S. (2002). Hierarchical Linear Models: Applications and Data Analysis Methods (2nd ed.). Sage.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian Data Analysis (3rd ed.). CRC Press.
