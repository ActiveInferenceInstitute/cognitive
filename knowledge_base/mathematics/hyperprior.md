---
title: Hyperprior
description: A prior distribution placed on the parameters of another prior, used in hierarchical Bayesian models to express uncertainty about scale or shape, and appearing in Active Inference as priors on precision.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - bayesian_statistics
  - priors
  - precision
semantic_relations:
  - type: relates
    links:
      - "[[prior_distribution]]"
      - "[[hierarchical_bayes]]"
      - "[[empirical_prior]]"
      - "[[precision_parameter]]"
      - "[[conjugate_priors]]"
---

# Hyperprior

## Overview
A hyperprior is a probability distribution over the hyperparameters of a prior. If observations $y$ depend on parameters $\theta$, and $\theta$ has a prior $p(\theta\mid\phi)$ governed by hyperparameters $\phi$, then a hyperprior $p(\phi)$ expresses uncertainty about $\phi$. The model is $p(y,\theta,\phi)=p(y\mid\theta)\,p(\theta\mid\phi)\,p(\phi)$, and the prior actually applied to $\theta$ is the marginal $p(\theta)=\int p(\theta\mid\phi)\,p(\phi)\,d\phi$. This is the fully Bayesian alternative to fixing $\phi$ by hand or estimating it from the data, as in empirical Bayes.

## Role and typical choices
- **Expressing uncertainty about scale**: a Gaussian prior on a coefficient has a variance or precision that is rarely known, and a hyperprior on it lets the data inform the amount of regularisation. Marginalising a Gaussian over a Gamma prior on its precision gives a Student-$t$ distribution, which has heavier tails.
- **Conjugate choices**: a Gamma hyperprior on the precision of a Gaussian, or an inverse-Gamma on its variance, preserves closed-form updates.
- **Weakly informative alternatives**: Gelman (2006) recommends half-Cauchy or half-$t$ priors for hierarchical scale parameters, since inverse-Gamma priors with small parameters can behave poorly when the scale is near zero.
- **Sensitivity**: in models with few groups, posterior inferences about scale are sensitive to the hyperprior, so sensitivity checks are routine.

A hyperprior is not a different kind of object from a prior; the label refers only to its position in the model hierarchy. The same distribution can be a prior in one description and a hyperprior in another.

## Relation to Active Inference
Hyperpriors appear in discrete-time models as priors over precision. In the formulation of Friston and colleagues (2013), the precision of the policy distribution is itself inferred, with a Gamma-distributed hyperprior whose posterior expectation changes as evidence accumulates, and the update of this quantity has been related to phasic dopamine. In continuous-time hierarchical models, hyperpriors on the log-precision of random fluctuations determine how strongly prediction errors are weighted. This use is part of the formal model specification. The proposed neurobiological reading of precision updates is a hypothesis that has been tested in specific tasks but is not established in general.

## Related concepts
- [[prior_distribution]] — the object of which the hyperprior is a prior
- [[hierarchical_bayes]] — the model class where hyperpriors appear
- [[empirical_prior]] — estimating hyperparameters instead of giving them a prior
- [[precision_parameter]] — the quantity most often given a hyperprior in Active Inference
- [[conjugate_priors]] — families that keep hyperparameter updates analytic

## References
- Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models. Bayesian Analysis, 1, 515-534.
- Friston, K., Schwartenbeck, P., FitzGerald, T., Moutoussis, M., Behrens, T., & Dolan, R. J. (2013). The anatomy of choice: Active inference and agency. Frontiers in Human Neuroscience, 7, 598.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian Data Analysis (3rd ed.). CRC Press.
