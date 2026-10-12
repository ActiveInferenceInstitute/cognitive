---
title: Adversarial Variational Bayes
description: A method that trains variational autoencoders with implicit posterior distributions by using a discriminator to estimate the divergence term, linking variational inference and generative adversarial networks.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - variational_inference
  - generative_models
  - deep_learning
semantic_relations:
  - type: relates
    links:
      - "[[variational_autoencoders]]"
      - "[[generative_adversarial_networks]]"
      - "[[variational_inference]]"
      - "[[variational_free_energy]]"
      - "[[kl_divergence]]"
      - "[[variational_methods]]"
---

# Adversarial Variational Bayes

## Overview
Adversarial Variational Bayes (AVB) is a technique introduced by Mescheder, Nowozin and Geiger (2017) for training latent-variable generative models when the approximate posterior is an implicit distribution: one that can be sampled but whose density cannot be evaluated. It replaces the analytic divergence term of a variational autoencoder (VAE) with an estimate from an auxiliary network trained adversarially, so that the posterior family is not restricted to tractable densities such as diagonal Gaussians.

## Construction
A VAE maximises the evidence lower bound (ELBO) for a model $p_\theta(x,z)=p_\theta(x|z)p(z)$ with approximate posterior $q_\phi(z|x)$ (Kingma and Welling, 2014):
$$\mathcal L(\theta,\phi)=\mathbb E_{p_D(x)}\Big[\mathbb E_{q_\phi(z|x)}\big[\log p_\theta(x|z)\big]-\mathrm{KL}\big(q_\phi(z|x)\,\|\,p(z)\big)\Big].$$
Equivalently, the integrand is $\mathbb E_{q_\phi}[\log p_\theta(x|z)-(\log q_\phi(z|x)-\log p(z))]$. In AVB, $q_\phi$ is defined by a generator $z=f_\phi(x,\epsilon)$ with noise $\epsilon$, which has no closed-form density, so the log-ratio $\log q_\phi(z|x)-\log p(z)$ cannot be computed directly. A discriminator $T_\psi(x,z)$ is trained to separate pairs $(x,z)$ with $z\sim q_\phi(z|x)$ from pairs with $z\sim p(z)$, with $x\sim p_D$ in both cases, by maximising
$$\mathbb E_{p_D(x)}\mathbb E_{q_\phi(z|x)}\log\sigma(T_\psi(x,z))+\mathbb E_{p_D(x)}\mathbb E_{p(z)}\log\big(1-\sigma(T_\psi(x,z))\big).$$
The optimal discriminator is $T^*(x,z)=\log q_\phi(z|x)-\log p(z)$, which is exactly the quantity needed. The encoder and decoder are then updated with $T_\psi$ substituted for the log-ratio. Mescheder et al. also describe a variant using adaptive contrast to improve behaviour when the posterior and prior are far apart.

## Properties and limits
With an optimal discriminator the objective equals the ELBO, so AVB is a form of maximum-likelihood training that inherits the interpretation of the bound. In practice the discriminator is only approximately optimal, so the bound is estimated and not guaranteed, and adversarial training can be unstable. The method links VAEs and generative adversarial networks (Goodfellow et al., 2014) by using adversarial density-ratio estimation inside a likelihood-based objective.

## Relation to Active Inference
Variational free energy in Active Inference is the negative of the ELBO, so methods that enlarge the family of approximate posteriors bear on how expressive free-energy minimisation can be. AVB is a machine learning technique that could in principle supply implicit posteriors for deep generative models used in deep Active Inference implementations, but it is not part of the standard Active Inference literature, which relies mostly on Gaussian and categorical posteriors.

## Related concepts
- [[variational_autoencoders]] — the model class that AVB generalises
- [[generative_adversarial_networks]] — source of the adversarial estimator
- [[variational_free_energy]] — the objective that the ELBO negates
- [[kl_divergence]] — divergence term estimated by the discriminator
- [[variational_inference]] — the approximation framework
- [[variational_methods]] — broader family of variational techniques

## References
- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. In Proceedings of the 2nd International Conference on Learning Representations.
- Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial nets. Advances in Neural Information Processing Systems, 27, 2672-2680.
- Mescheder, L., Nowozin, S., & Geiger, A. (2017). Adversarial variational Bayes: Unifying variational autoencoders and generative adversarial networks. Proceedings of the 34th International Conference on Machine Learning, PMLR 70, 2391-2400.
