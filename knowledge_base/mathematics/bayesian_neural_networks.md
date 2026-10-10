---
title: Bayesian Neural Networks
description: Neural networks with probability distributions over their weights, trained by approximate posterior inference, giving predictive uncertainty and a formal link to variational free energy minimisation.
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - neural_networks
  - variational_inference
  - uncertainty
semantic_relations:
  - type: relates
    links:
      - "[[neural_network_theory]]"
      - "[[variational_inference]]"
      - "[[uncertainty_quantification]]"
      - "[[probabilistic_models]]"
---

# Bayesian Neural Networks

## Overview
A Bayesian neural network (BNN) treats the weights $w$ of a network as random variables with a prior $p(w)$ and infers a posterior $p(w\mid\mathcal{D})$ from data $\mathcal{D}=\{(x_i,y_i)\}$. Prediction averages over that posterior instead of using one weight setting, so the model reports epistemic uncertainty about its parameters in addition to noise in the data.

## Formalism
Bayes' rule gives $p(w\mid\mathcal{D})\propto p(\mathcal{D}\mid w)\,p(w)$, and the posterior predictive is
$$p(y^*\mid x^*,\mathcal{D})=\int p(y^*\mid x^*,w)\,p(w\mid\mathcal{D})\,dw.$$
The posterior is intractable for realistic networks, so approximations are used:
- **Variational inference.** Choose a family $q_\phi(w)$ (often a factorised Gaussian) and maximise the evidence lower bound $\mathcal{L}=\mathbb{E}_{q}[\log p(\mathcal{D}\mid w)]-\mathrm{KL}(q_\phi(w)\,\|\,p(w))$. Gradients use the reparameterisation trick (Bayes by backprop).
- **MCMC.** Hamiltonian or Langevin sampling gives asymptotically exact samples at high cost ([[langevin_dynamics]]).
- **Laplace approximation.** A Gaussian centred on a trained mode with covariance from the inverse Hessian.
- **Ensembles and dropout.** Deep ensembles and Monte Carlo dropout give practical uncertainty estimates; the interpretation of dropout as approximate inference is argued by Gal and Ghahramani (2016) and debated.

Quality depends on the prior and the approximating family. Mean-field Gaussians underestimate posterior correlations, and weight-space priors are hard to interpret.

## Role in Active Inference
The ELBO above is variational free energy with weights as the latent variables: accuracy is the expected log-likelihood and complexity is the KL to the prior. Training a BNN by variational inference is therefore the same computation as perception or learning in an Active Inference model, applied to parameters. Free-energy treatments of learning make this explicit, treating synaptic parameters as slowly varying hidden causes that are inferred. Neural-network generative models inside Active Inference (deep Active Inference) can use BNN-style uncertainty to drive the epistemic term in expected free energy. This is an established formal correspondence; it does not imply that cortex implements a BNN.

## Related concepts
- [[variational_inference]] — the approximation method most commonly used for BNN posteriors
- [[variational_free_energy]] — the objective that coincides with the negative ELBO
- [[neural_network_theory]] — architectural background for the networks being made Bayesian
- [[uncertainty_quantification]] — the aim of reporting predictive uncertainty
- [[bayesian_statistics]] — the inferential framework applied to weights
- [[probabilistic_models]] — broader family of models with distributions over parameters
