---
title: Hierarchical Bayes
description: Probability models with parameters organised in levels, where group-level parameters share a common distribution, giving partial pooling, shrinkage and the model structure behind hierarchical Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - bayesian_statistics
  - hierarchical_models
  - partial_pooling
semantic_relations:
  - type: relates
    links:
      - "[[hyperprior]]"
      - "[[empirical_prior]]"
      - "[[bayesian_statistics]]"
      - "[[hierarchical_models]]"
      - "[[predictive_coding]]"
---

# Hierarchical Bayes

## Overview
A hierarchical Bayesian model places parameters in layers. Data for group $j$ are generated from parameters $\theta_j$, those parameters are drawn from a population distribution with hyperparameters $\phi$, and $\phi$ has its own (hyper)prior:
$$
y_{ij}\sim p(y\mid\theta_j),\qquad \theta_j\sim p(\theta\mid\phi),\qquad \phi\sim p(\phi).
$$
The joint posterior over $(\theta_1,\dots,\theta_J,\phi)$ is obtained by Bayes' rule, so information flows between groups through $\phi$. Because the structure matches data with a natural grouping, such as students within schools, patients within hospitals or trials within participants, these models are also called multilevel models (Gelman & Hill, 2007).

## Partial pooling
Two extremes bracket the hierarchical model. Complete pooling assumes all groups share one parameter, and no pooling fits each group independently. Hierarchical modelling lies between them: each group estimate is a precision-weighted compromise between its own data and the population mean, so groups with little data are shrunk strongly toward the mean and groups with abundant data are shrunk little. This improves estimates for small groups and reduces overfitting. Rubin's (1981) analysis of parallel randomised experiments in eight schools is a standard worked example. The population variance is the key hyperparameter, and its prior needs care when the number of groups is small. Inference is usually carried out with Markov chain Monte Carlo or variational methods.

## Relation to Active Inference
The relation is structural and central. Cortical hierarchies in predictive coding are described as hierarchical generative models in which the causes at each level act as priors for the level below, so the higher level supplies empirical priors for lower ones (Friston, 2005). Inference then amounts to hierarchical Bayesian inference carried out by message passing, with prediction errors as the messages. Deep temporal models extend this to nested timescales. How closely brains implement exactly this scheme remains an empirical question, but the mathematical description of the generative model is hierarchical Bayes.

## Related concepts
- [[hyperprior]] — the prior placed on population-level parameters
- [[empirical_prior]] — priors informed by higher levels or by estimated hyperparameters
- [[bayesian_statistics]] — the inferential framework for these models
- [[hierarchical_models]] — the cognitive and neural counterpart
- [[hierarchical_processing]] — hierarchical organisation in the brain
- [[predictive_coding]] — a neural scheme for hierarchical inference

## References
- Rubin, D. B. (1981). Estimation in parallel randomized experiments. Journal of Educational Statistics, 6, 377-401.
- Gelman, A., & Hill, J. (2007). Data Analysis Using Regression and Multilevel/Hierarchical Models. Cambridge University Press.
- Friston, K. (2005). A theory of cortical responses. Philosophical Transactions of the Royal Society B, 360, 815-836.
