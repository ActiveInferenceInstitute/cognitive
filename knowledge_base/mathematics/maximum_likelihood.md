---
title: Maximum Likelihood
description: "The maximum likelihood estimator, its Gaussian closed form, asymptotic properties, relation to KL divergence and MAP estimation, and its place beneath variational free energy."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - statistics
  - estimation
  - inference
semantic_relations:
  - type: relates
    links:
      - "[[fisher_information]]"
      - "[[kl_divergence]]"
      - "[[variational_free_energy]]"
      - "[[statistical_foundations]]"
---

# Maximum Likelihood

## Overview
Maximum likelihood estimation (MLE) chooses the parameter value under which the observed data are most probable: $\hat\theta=\arg\max_\theta \ln p(x_{1:n}\mid\theta)$. It is a point estimator with no prior, and the usual frequentist default for fitting parametric models.

## Properties and relations
**Gaussian example.** For i.i.d. $x_i\sim\mathcal N(\mu,\sigma^2)$, the MLE is $\hat\mu=\bar x$ and $\hat\sigma^2=\frac1n\sum(x_i-\bar x)^2$. The latter is biased, with expectation $\frac{n-1}{n}\sigma^2$, which shows that MLE does not guarantee unbiasedness.

**Asymptotics.** Under regularity conditions, $\hat\theta$ is consistent, and $\sqrt n(\hat\theta-\theta_0)\to\mathcal N(0,\,I(\theta_0)^{-1})$ where $I$ is the Fisher information ([[fisher_information]]). The estimator attains the Cramer-Rao bound asymptotically.

**KL view.** Maximising the average log-likelihood is equivalent to minimising $D_{KL}(\hat p_{\text{data}}\,\|\,p_\theta)$, where $\hat p_{\text{data}}$ is the empirical distribution. Cross-entropy training of classifiers is MLE.

**Relation to MAP and Bayes.** Maximum a posteriori estimation adds $\ln p(\theta)$ to the objective, and equals MLE under a flat prior. A full Bayesian treatment keeps the whole posterior rather than its mode.

**Latent variables.** When $p(x\mid\theta)=\int p(x,z\mid\theta)\,dz$ is intractable, expectation-maximisation maximises a lower bound, using the identity $\ln p(x\mid\theta)=\mathcal L(q,\theta)+D_{KL}(q(z)\,\|\,p(z\mid x,\theta))$. The bound $\mathcal L$ is the negative variational free energy.

## Role in Active Inference
This is a derivation-level link. The log evidence $\ln p(o\mid m)$ that variational free energy bounds is the marginal likelihood, and minimising $F$ with respect to model parameters is variational EM, an approximate type-II maximum likelihood (or MAP, when parameters have priors). Parameter updating in [[parameter_learning]] goes further than point estimates by keeping Dirichlet posteriors. Where the literature speaks of "maximum likelihood decoding" of neural populations ([[neural_coding]]), it uses the estimator in the narrow sense above.

## Related concepts
- [[bayes_theorem]] — the prior-weighted counterpart to likelihood
- [[variational_free_energy]] — the bound maximised in variational EM
- [[information_criteria]] — penalised maximum likelihood for model comparison
- [[kl_divergence]] — the divergence that MLE minimises
- [[parameter_learning]] — Bayesian treatment of parameters
