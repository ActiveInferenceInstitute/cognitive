---
title: Gaussian Processes
description: Distributions over functions defined by a mean and covariance kernel, how they give closed-form regression with uncertainty, and their use as surrogate models in Bayesian optimisation and information-seeking.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - nonparametric_models
  - bayesian_statistics
  - regression
semantic_relations:
  - type: relates
    links:
      - "[[bayesian_inference]]"
      - "[[bayesian_optimization]]"
      - "[[epistemic_value]]"
      - "[[probabilistic_models]]"
      - "[[uncertainty_estimation]]"
---

# Gaussian Processes

## Overview
A Gaussian process (GP) is a collection of random variables, any finite subset of which has a joint Gaussian distribution. It is used as a prior over functions $f:\mathcal{X}\to\mathbb{R}$ and is specified by a mean function $m(x)$ and a covariance function, or kernel, $k(x,x')$, written $f\sim\mathcal{GP}(m,k)$. Because the model places a distribution directly on functions rather than on a fixed set of parameters, it is called nonparametric.

## Regression
Suppose noisy observations $y_i=f(x_i)+\varepsilon_i$ with $\varepsilon_i\sim\mathcal{N}(0,\sigma^2)$ at inputs $X$. The joint distribution of $y$ and the function value $f_*$ at a new input $x_*$ is Gaussian, so conditioning gives a Gaussian posterior with closed-form moments:

$$\mu_*=k_*^{\top}(K+\sigma^2I)^{-1}y,\qquad \sigma_*^2=k(x_*,x_*)-k_*^{\top}(K+\sigma^2I)^{-1}k_*,$$

where $K_{ij}=k(x_i,x_j)$ and $k_*$ is the vector of covariances between $x_*$ and the training inputs. The predictive variance grows away from data, so the model reports its own uncertainty. The kernel encodes assumptions such as smoothness and length scale; common choices are the squared exponential and Matern kernels. Hyperparameters can be fit by maximising the marginal likelihood. Exact inference costs $O(n^3)$ in the number of observations, which motivates sparse and inducing-point approximations.

## Relation to Active Inference and information seeking
The link is practical and analogical rather than a derivation. GPs are a natural surrogate when an agent must reason about an unknown function and its uncertainty. In Bayesian optimisation the predictive mean and variance are combined into an acquisition function that trades off exploiting good regions against exploring uncertain ones (see [[bayesian_optimization]]). This resembles the split of expected free energy into extrinsic and epistemic terms, and with a GP the expected information gain about the function has a closed form, since entropy of a Gaussian depends only on its covariance (see [[epistemic_value]]). GPs have been used as generative models in some Active Inference and curiosity-driven control work, but the standard discrete and continuous-time formulations do not require them. Claims that the brain implements GP inference are hypotheses, not established results.

## Related concepts
- [[bayesian_inference]] — GP regression is exact Bayesian inference over functions
- [[bayesian_optimization]] — uses GP predictions to choose where to sample
- [[epistemic_value]] — information gain about unknown quantities
- [[probabilistic_models]] — the broader family of models with explicit uncertainty
- [[uncertainty_estimation]] — GP predictive variance as an uncertainty measure

## References
- Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
