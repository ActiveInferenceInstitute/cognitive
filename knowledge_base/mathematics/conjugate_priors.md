---
title: Conjugate Priors
description: Prior families that keep the posterior in the same family as the prior, giving closed-form Bayesian updates, with standard examples and their role in discrete and Gaussian Active Inference models.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - bayesian_statistics
  - exponential_families
  - priors
semantic_relations:
  - type: relates
    links:
      - "[[bayesian_inference]]"
      - "[[exponential_family]]"
      - "[[prior_beliefs]]"
      - "[[statistical_foundations]]"
      - "[[variational_inference]]"
---

# Conjugate Priors

## Overview
A family of prior distributions $\mathcal{P}$ is conjugate to a likelihood $p(x\mid\theta)$ if, for every prior $p(\theta)\in\mathcal{P}$, the posterior $p(\theta\mid x)\propto p(x\mid\theta)\,p(\theta)$ also lies in $\mathcal{P}$. Updating then reduces to changing a small number of parameters of the prior, so no integration is needed to obtain the posterior or the marginal likelihood. The idea goes back to Raiffa and Schlaifer, who studied it for decision analysis.

## Standard pairs
- **Beta and Bernoulli or binomial.** A $\mathrm{Beta}(\alpha,\beta)$ prior on a success probability becomes $\mathrm{Beta}(\alpha+k,\beta+n-k)$ after $k$ successes in $n$ trials.
- **Dirichlet and categorical or multinomial.** A $\mathrm{Dir}(\boldsymbol\alpha)$ prior becomes $\mathrm{Dir}(\boldsymbol\alpha+\mathbf{c})$ after observing category counts $\mathbf{c}$. The beta case is the two-category special case.
- **Gaussian and Gaussian.** For a known observation variance, a Gaussian prior on the mean yields a Gaussian posterior whose precision is the sum of prior and data precisions.
- **Gamma and Poisson, or gamma and Gaussian precision.** A gamma prior on a rate or precision stays gamma.

The general explanation is that every likelihood in the [[exponential_family]] has a conjugate prior of the same functional form, with hyperparameters that behave like pseudo-observations: a prior count plus the observed sufficient statistics. Diaconis and Ylvisaker (1979) studied these conjugate families in detail, including the property that the posterior mean of the mean parameter is linear in the data.

## Role in Active Inference
Conjugacy is central to the discrete-state formulation. The likelihood and transition matrices are columns of categorical distributions, and Dirichlet priors over them make parameter learning a count-accumulation rule: concentration parameters are incremented in proportion to the posterior beliefs about co-occurring states and outcomes. This is how Active Inference implementations describe learning of the A and B arrays, and it is also the form that habit learning over policies takes. Conjugacy keeps the approximate posteriors used in [[variational_inference]] in a known family, so mean-field updates have closed forms. This link is a standard modelling convenience rather than a claim about neural implementation. Conjugate choices are made because they are tractable, and they restrict the prior beliefs that can be expressed; non-conjugate models need sampling or numerical approximation.

## Related concepts
- [[bayesian_inference]] — the updating scheme that conjugacy simplifies
- [[exponential_family]] — the setting in which conjugate priors exist generally
- [[prior_beliefs]] — how prior knowledge enters inference
- [[variational_inference]] — approximate posteriors with conjugate factor updates
- [[monte_carlo_methods]] — the alternative when no conjugate form exists

## References
- Raiffa, H., & Schlaifer, R. (1961). Applied Statistical Decision Theory. Harvard Business School.
- Diaconis, P., & Ylvisaker, D. (1979). Conjugate priors for exponential families. Annals of Statistics, 7, 269-281.
- Gelman, A., et al. (2013). Bayesian Data Analysis (3rd ed.). Chapman and Hall/CRC.
