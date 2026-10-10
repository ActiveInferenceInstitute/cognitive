---
title: Bayesian Statistics
description: The statistical framework that represents unknown quantities by probability distributions, updated by Bayes' rule, covering priors, posteriors, conjugacy, model evidence and its role as the basis of Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - statistics
  - bayesian_inference
  - probability
semantic_relations:
  - type: relates
    links:
      - "[[bayes_theorem]]"
      - "[[probability_theory]]"
      - "[[statistical_modeling]]"
      - "[[bayesian_models]]"
---

# Bayesian Statistics

## Overview
Bayesian statistics interprets probability as a degree of belief and treats unknown parameters, hidden states and models as random variables. Inference is the conditioning of a joint model on observed data. This differs from frequentist practice, which treats parameters as fixed and attaches probabilities only to repeatable data-generating procedures.

## Core ingredients
- **Prior** $p(\theta)$: beliefs before seeing data.
- **Likelihood** $p(y\mid\theta)$: probability of the data under each parameter value.
- **Posterior** $p(\theta\mid y)=p(y\mid\theta)p(\theta)/p(y)$, from [[bayes_theorem]].
- **Marginal likelihood (evidence)** $p(y)=\int p(y\mid\theta)p(\theta)\,d\theta$, used to compare models. The ratio of two evidences is a Bayes factor.
- **Posterior predictive** $p(\tilde y\mid y)=\int p(\tilde y\mid\theta)p(\theta\mid y)\,d\theta$.
- **Credible intervals** summarise the posterior directly: a 95% credible interval contains 95% of posterior mass.

**Conjugacy.** When the prior and posterior share a family, updates are closed-form. For a Beta$(a,b)$ prior on a success probability and $k$ successes in $n$ trials, the posterior is Beta$(a+k,\,b+n-k)$. With a uniform prior ($a=b=1$) and 7 successes in 10 trials the posterior is Beta$(8,4)$ with mean $2/3$. Gaussian-Gaussian and Dirichlet-categorical pairs behave the same way, and the exponential family is the general setting ([[exponential_family]]).

**Computation.** Non-conjugate models require approximation: Markov chain Monte Carlo, importance sampling ([[importance_sampling]]), variational inference ([[variational_inference]]) or Laplace approximations.

## Role in Active Inference
Active Inference is Bayesian inference over a generative model with an added decision rule. Perception is posterior inference over hidden states, learning is posterior inference over parameters, and model selection uses evidence, approximated by variational free energy, which bounds the log evidence from below. The distinctive step is not statistical: agents also act so that future observations conform to preferred outcomes. Priors in this setting often carry preferences, so the prior is a modelling choice with behavioural consequences, and not only a statement of ignorance.

## Related concepts
- [[bayes_theorem]] — the update rule underlying everything on this page
- [[conditional_probability]] — the definition on which Bayes' rule rests
- [[bayesian_inference]] — the cognitive-science treatment of the same principle
- [[variational_inference]] — tractable approximation to posteriors
- [[information_criteria]] — penalised-fit alternatives to evidence
- [[probability_theory]] — measure-theoretic foundations

## References
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. and Rubin, D. B. (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press.
