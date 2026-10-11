---
title: Bayesian Deep Learning
description: Neural networks treated probabilistically, with distributions over weights or predictions, giving uncertainty estimates through approximate inference methods such as variational inference, dropout, ensembles and Laplace approximations.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - deep_learning
  - uncertainty
  - approximate_inference
semantic_relations:
  - type: relates
    links:
      - "[[bayesian_neural_networks]]"
      - "[[deep_learning]]"
      - "[[variational_inference]]"
      - "[[epistemic_uncertainty]]"
      - "[[uncertainty_quantification]]"
---

# Bayesian Deep Learning

## Overview
Standard neural network training returns a single weight vector $\hat w$. Bayesian deep learning instead places a prior $p(w)$ on the weights and seeks the posterior $p(w\mid D)\propto p(D\mid w)\,p(w)$, so that predictions average over plausible networks: $p(y^{*}\mid x^{*},D)=\int p(y^{*}\mid x^{*},w)\,p(w\mid D)\,dw$. The spread of this predictive distribution separates regions where the data constrain the model from regions where they do not, which supports calibrated confidence, out-of-distribution detection and information-seeking data collection. The early formulation is due to MacKay (1992) and Neal (1996).

## Approximation methods
Exact posteriors over millions of weights are intractable, so several approximations are used:
- **Variational inference**: fit a tractable family, often a factorised Gaussian, by minimising the KL divergence to the posterior, which maximises the evidence lower bound (for example Bayes by Backprop).
- **Monte Carlo dropout**: keeping dropout active at test time can be interpreted as approximate variational inference (Gal & Ghahramani, 2016).
- **Deep ensembles**: train several networks from different initialisations and average. This is not strictly Bayesian but often gives well-calibrated uncertainty.
- **Laplace approximation**: fit a Gaussian around a trained solution using the curvature of the loss.
- **Sampling methods**: stochastic-gradient variants of Hamiltonian Monte Carlo and Langevin dynamics.

The quality of the approximation and the choice of prior over weights both affect the resulting uncertainty, and cheap approximations tend to underestimate it.

## Relation to Active Inference
The link is conceptual and methodological. Active Inference is itself a variational Bayesian scheme with a generative model, and deep generative models can be used to parameterise its likelihood, transition or policy distributions, with training done by the same free-energy objective. The Bayesian brain hypothesis supplies the motivation for treating learned parameters as uncertain quantities. Epistemic uncertainty about parameters is also what makes information-seeking policies meaningful, since an agent can only expect to learn about parameters it holds uncertain beliefs over. Deep-learning implementations of Active Inference are an active research area and vary widely in how Bayesian they actually are.

## Related concepts
- [[bayesian_neural_networks]] — networks with distributions over weights
- [[variational_inference]] — the main approximation technique
- [[epistemic_uncertainty]] — reducible uncertainty about model parameters
- [[bayesian_brain]] — the neuroscientific hypothesis motivating probabilistic models
- [[uncertainty_quantification]] — the wider aim of reporting reliable uncertainty

## References
- MacKay, D. J. C. (1992). A practical Bayesian framework for backpropagation networks. Neural Computation, 4, 448-472.
- Neal, R. M. (1996). Bayesian Learning for Neural Networks. Springer.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. Proceedings of the 33rd International Conference on Machine Learning, PMLR 48, 1050-1059.
