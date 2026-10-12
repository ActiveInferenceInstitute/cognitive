---
title: Likelihood Function
description: The probability of observed data viewed as a function of model parameters, its distinction from a probability distribution over parameters, and its role in estimation, Bayes theorem and observation models.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistics
  - bayesian_inference
  - estimation
semantic_relations:
  - type: relates
    links:
      - "[[bayes_theorem]]"
      - "[[prior_distribution]]"
      - "[[maximum_likelihood]]"
      - "[[fisher_information]]"
      - "[[observation_model]]"
wikidata: Q45284
---

# Likelihood Function

## Overview
For a statistical model $p(y\mid\theta)$ with parameters $\theta$, the likelihood function is the same expression read with the data $y$ held fixed and the parameters varying:
$$\mathcal{L}(\theta;y)=p(y\mid\theta).$$
It is not a probability distribution over $\theta$. It need not integrate to one over the parameter space, and it is defined only up to a multiplicative constant that does not depend on $\theta$. Its ratios between parameter values measure how strongly the data favour one value over another.

## Use in inference
- **Maximum likelihood**: the estimate $\hat\theta=\arg\max_\theta \ln\mathcal{L}(\theta;y)$ is usually computed from the log-likelihood, which turns products over independent observations into sums.
- **Curvature**: the curvature of the log-likelihood at its maximum determines the precision of the estimate. The expected curvature is the Fisher information, which sets the Cramér-Rao bound on estimator variance.
- **Likelihood ratio**: comparing the likelihood at two parameter values or under two models underlies classical hypothesis tests and Bayes factors.
- **Likelihood principle**: given the model, all the evidence the data contain about $\theta$ is carried by the likelihood function. Frequentist procedures that depend on unobserved outcomes violate this principle.

## Bayesian role
Bayes theorem combines the likelihood with a prior, $p(\theta\mid y)\propto p(y\mid\theta)\,p(\theta)$. In graphical models and factor graphs, the likelihood is the factor linking observed nodes to their parents, and it is the term updated by observing a new data point. A likelihood can also be specified for latent states, as in $p(y\mid x,\theta)$, where $x$ are hidden variables and $\theta$ are parameters.

## Relation to Active Inference
In discrete-state Active Inference the likelihood is the observation model: a matrix, conventionally called $A$, whose entries give the probability of each outcome under each hidden state. Its columns are categorical distributions. Its entropy is the quantity called ambiguity in the expected free energy, so states with a noisy mapping to outcomes are penalised. In continuous-time formulations the likelihood is a Gaussian density centred on a function of the hidden states, and its precision weights sensory prediction errors. The link is a direct part of the formalism.

## Related concepts
- [[bayes_theorem]] — combines likelihood and prior into a posterior
- [[prior_distribution]] — the other factor in the posterior
- [[maximum_likelihood]] — estimation by maximising the likelihood
- [[fisher_information]] — expected curvature of the log-likelihood
- [[observation_model]] — the likelihood as used for agents
- [[A_matrix]] — the discrete likelihood mapping in Active Inference

## References
- Fisher, R. A. (1922). On the mathematical foundations of theoretical statistics. Philosophical Transactions of the Royal Society of London A, 222, 309-368.
- Casella, G., & Berger, R. L. (2002). Statistical Inference (2nd ed.). Duxbury.
- Berger, J. O., & Wolpert, R. L. (1988). The Likelihood Principle (2nd ed.). Institute of Mathematical Statistics.
