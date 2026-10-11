---
title: Latent Variable Models
description: Probabilistic models that explain observed data through unobserved variables, with standard examples, inference by marginalisation, and the identification of hidden states in Active Inference generative models.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - probabilistic_models
  - generative_models
  - inference
semantic_relations:
  - type: relates
    links:
      - "[[bayesian_generative_models]]"
      - "[[generative_models]]"
      - "[[expectation_maximization]]"
      - "[[variational_inference]]"
      - "[[variational_autoencoders]]"
---

# Latent Variable Models

## Overview
A latent variable model describes observed variables $x$ through unobserved variables $z$ that are never measured directly. The model specifies a prior $p(z)$ and a conditional $p(x\mid z)$, and the distribution of the data is obtained by marginalisation:
$$p(x)=\int p(x\mid z)\,p(z)\,dz.$$
Latent structure lets a small number of hidden causes account for correlations among many observations.

## Standard examples
- **Mixture models**: $z$ is a discrete component label, and $p(x)=\sum_k \pi_k\,p(x\mid z=k)$.
- **Factor analysis and probabilistic principal component analysis**: $z$ is a low-dimensional Gaussian vector and $x$ is a linear map of $z$ plus Gaussian noise.
- **Hidden Markov models and state-space models**: $z_t$ evolves as a Markov chain and each $x_t$ depends only on $z_t$.
- **Variational autoencoders**: $p(x\mid z)$ is parameterised by a neural network.

## Inference and learning
Two tasks recur: inferring the posterior $p(z\mid x)=p(x\mid z)p(z)/p(x)$, and estimating the model parameters. The marginal likelihood $p(x)$ is often intractable, which motivates the expectation-maximisation algorithm, variational inference and sampling. Latent variable models can be unidentifiable: different parameter settings, such as permuted component labels or rotated factors, may give identical data distributions, so conclusions about $z$ must be stated up to the relevant symmetry.

## Relation to Active Inference
The link is structural and exact. An Active Inference agent is equipped with a generative model in which hidden states cause observations, and perception is posterior inference over those latent variables. In the discrete formulation, hidden states, the likelihood mapping from states to outcomes and the transition dynamics define a hidden Markov model extended with control. The agent's beliefs are approximate posteriors that minimise variational free energy, the same bound used to train latent variable models in machine learning. Structure learning, the question of how many latent states or factors a model should contain, is an active research topic.

## Related concepts
- [[bayesian_generative_models]] — the general class of models that latent variable models instantiate
- [[generative_models]] — the agent's internal model of how observations arise
- [[expectation_maximization]] — a classical algorithm for fitting latent variable models
- [[variational_autoencoders]] — a deep latent variable model trained by a free energy bound
- [[likelihood_function]] — the conditional that links latent variables to data
- [[prior_distribution]] — the distribution assigned to the latent variables

## References
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. Journal of the Royal Statistical Society: Series B, 39, 1-38.
