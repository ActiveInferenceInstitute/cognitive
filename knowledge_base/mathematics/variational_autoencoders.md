---
title: Variational Autoencoders
description: "Latent-variable generative models trained by maximising the evidence lower bound with amortised encoders and the reparameterisation trick, and their close formal match to variational free energy."
type: mathematical_concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - deep_learning
  - generative_models
  - variational_inference
semantic_relations:
  - type: relates
    links:
      - "[[variational_inference]]"
      - "[[variational_free_energy]]"
      - "[[generative_models]]"
      - "[[deep_learning]]"
---

# Variational Autoencoders

## Overview
A variational autoencoder (VAE) is a deep latent-variable model $p_\theta(x,z)=p_\theta(x\mid z)p(z)$ paired with an inference network $q_\phi(z\mid x)$ that approximates the intractable posterior. Both networks are trained jointly by stochastic gradient ascent on one objective, the evidence lower bound (ELBO).

## Formalism
**Objective.** For each data point,
$$\ln p_\theta(x)\ \ge\ \mathcal L(\theta,\phi;x)=\mathbb E_{q_\phi(z\mid x)}[\ln p_\theta(x\mid z)]-D_{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big),$$
with gap $D_{KL}(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x))$. The first term is reconstruction accuracy; the second regularises the encoder toward the prior.

**Reparameterisation.** For Gaussian $q_\phi=\mathcal N(\mu_\phi(x),\sigma_\phi^2(x))$ write $z=\mu_\phi+\sigma_\phi\odot\epsilon$ with $\epsilon\sim\mathcal N(0,I)$, so gradients pass through the sampling step. For a standard-normal prior the KL has the closed form $\tfrac12\sum_j(\sigma_j^2+\mu_j^2-1-\ln\sigma_j^2)$, which I checked against a Monte Carlo estimate (0.638 versus 0.639 for $\mu=0.8,\sigma=0.5$).

**Amortisation.** One network evaluates the approximate posterior for all inputs, rather than optimising separate variational parameters per data point. This trades some accuracy (the amortisation gap) for speed.

**Variants.** The $\beta$-VAE reweights the KL term; hierarchical VAEs stack latent layers; discrete and vector-quantised versions replace Gaussian latents.

## Role in Active Inference
The ELBO is the negative variational free energy $F=D_{KL}(q\|p(z))-\mathbb E_q[\ln p(x\mid z)]$, with the same complexity-minus-accuracy decomposition ([[variational_free_energy]]). A VAE is therefore a perceptual-inference system for a static generative model with an amortised recognition density. Deep Active Inference agents use VAE components as observation models and state encoders, adding transition models and expected free energy to choose actions (Millidge 2020; Fountas et al. 2020). The difference is in what the framework claims: VAEs optimise the bound as an engineering objective, whereas Active Inference interprets the same quantity as the quantity organisms minimise, and adds action through expected free energy.

## Related concepts
- [[variational_inference]] — the approximation scheme VAEs amortise
- [[variational_bounds]] — the ELBO and its tightness
- [[generative_models]] — models of how data arise from latents
- [[deep_learning]] — networks used for encoder and decoder
- [[neural_architectures]] — the parameterisations in use

## References
- Kingma, D. P. and Welling, M. (2014). Auto-encoding variational Bayes. ICLR.
- Rezende, D. J., Mohamed, S., Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models. ICML.
- Fountas, Z., Sajid, N., Mediano, P., Friston, K. (2020). Deep active inference agents using Monte-Carlo methods. NeurIPS.
- Millidge, B. (2020). Deep active inference as variational policy gradients. Journal of Mathematical Psychology.
