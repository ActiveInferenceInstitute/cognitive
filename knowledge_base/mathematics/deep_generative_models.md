---
title: Deep Generative Models
description: Deep generative models use neural networks to learn a probability distribution over data and sample from it, including variational autoencoders, adversarial networks, normalizing flows, autoregressive models and diffusion models.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - deep_learning
  - generative_models
  - probabilistic_modeling
semantic_relations:
  - type: relates
    links:
      - "[[variational_autoencoders]]"
      - "[[generative_adversarial_networks]]"
      - "[[normalizing_flows]]"
      - "[[diffusion_models]]"
      - "[[generative_model]]"
---

# Deep Generative Models

## Overview
A deep generative model is a neural network trained to represent a probability distribution $p_\theta(x)$ close to the distribution of the data, so that new samples can be drawn and, in some families, likelihoods evaluated. The models differ chiefly in how they make the distribution tractable to train and sample from.

## Main families
- **Latent-variable models with amortised inference.** A variational autoencoder defines $p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz$ and trains an encoder $q_\phi(z\mid x)$ by maximising the evidence lower bound
$$\log p_\theta(x)\;\ge\;\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]-\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big).$$
The same quantity is the negative variational free energy.
- **Adversarial models.** A generator is trained against a discriminator in a two-player game. The generator's density is implicit, so likelihoods are not available, and training can be unstable.
- **Normalizing flows.** Invertible maps give exact likelihoods through the change-of-variables formula, at the price of architectural constraints or costly Jacobian terms.
- **Autoregressive models.** The joint density is factorised as $\prod_i p(x_i\mid x_{<i})$, which gives exact likelihoods and sequential sampling.
- **Diffusion models.** Data are gradually corrupted by noise, and a network learns to reverse the process, which can be viewed as score estimation.

## Trade-offs
Evaluation of these models is not straightforward. Likelihood-based comparison is possible only for families with tractable or boundable densities, and sample quality is assessed with separate metrics. Models may memorise training examples, or cover only part of the data distribution. Training objectives therefore reflect compromises between expressiveness, tractability, sample quality and speed.

## Relation to Active Inference
In Active Inference an agent is equipped with a generative model of how hidden states cause observations, and perception and learning are inference and parameter updates under that model. Deep generative models supply flexible function approximators for the likelihood and transition components when the state space is too large for tabular matrices. This is an active research programme under the name of deep active inference. The variational autoencoder bound and the free energy used in Active Inference are the same quantity, but the machine-learning use is usually pure density modelling, without action or expected free energy.

## Related concepts
- [[variational_autoencoders]] — latent-variable models trained with the evidence bound
- [[generative_adversarial_networks]] — implicit models trained by a two-player game
- [[normalizing_flows]] — exact-likelihood models based on invertible maps
- [[diffusion_models]] — models built on iterative denoising
- [[variational_free_energy]] — the objective shared with Active Inference
- [[variational_methods]] — the approximation techniques used in training

## References
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. International Conference on Learning Representations.
- Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial nets. Advances in Neural Information Processing Systems, 27.
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33.
