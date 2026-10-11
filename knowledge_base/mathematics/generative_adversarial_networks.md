---
title: Generative Adversarial Networks
description: A framework in which a generator and a discriminator network are trained against each other in a minimax game to model a data distribution without an explicit likelihood.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - deep_learning
  - generative_models
  - game_theory
semantic_relations:
  - type: relates
    links:
      - "[[variational_autoencoders]]"
      - "[[jensen_shannon_divergence]]"
      - "[[generative_models]]"
      - "[[neural_networks]]"
      - "[[game_theory]]"
---

# Generative Adversarial Networks

## Overview
A generative adversarial network (GAN) pairs two neural networks. The generator $G$ maps random noise $z\sim p_z$ to samples $G(z)$, and the discriminator $D$ outputs the probability that its input came from the data and not from the generator. The two are trained simultaneously: $D$ tries to tell real from generated samples, and $G$ tries to fool $D$. The model is implicit, since it can produce samples but does not provide a tractable density.

## Objective
In the original formulation the networks play the minimax game
$$\min_G \max_D \; \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))].$$
For a fixed generator the optimal discriminator is $D^*(x)=p_{\text{data}}(x)/(p_{\text{data}}(x)+p_g(x))$, where $p_g$ is the distribution of generated samples. Substituting $D^*$ gives a value of $-\log 4 + 2\,\mathrm{JSD}(p_{\text{data}}\,\|\,p_g)$, so the generator minimises the Jensen-Shannon divergence to the data distribution, with a global minimum when $p_g=p_{\text{data}}$. In practice training alternates gradient steps on the two networks, and the guarantee holds only in an idealised setting with unlimited capacity.

## Practical issues
- **Instability**: simultaneous gradient updates in a two-player game need not converge, and training can oscillate.
- **Mode collapse**: the generator may cover only a few modes of the data.
- **Vanishing gradients**: when the discriminator becomes too accurate the generator receives little signal, which motivated alternative losses such as those based on the Wasserstein distance.

## Relation to Active Inference
There is no derivation connecting GANs to the free energy principle. The contrast is nonetheless instructive. A GAN learns a generative model by comparing samples through an adversarial critic, while Active Inference and variational autoencoders learn by bounding model evidence, which requires an explicit likelihood and a recognition density. The generator and discriminator are loosely analogous to a generative process and a classifier of its outputs, but this is an analogy at the level of architecture. Adversarial ideas appear in some work on predictive processing, as models trained to discriminate real from self-generated input, yet these are proposals and not established results.

## Related concepts
- [[variational_autoencoders]] — a likelihood-based generative model with a recognition network
- [[jensen_shannon_divergence]] — the divergence minimised at the optimal discriminator
- [[game_theory]] — the minimax structure of the training objective
- [[deep_learning]] — the setting in which GANs are built
- [[energy_based_models]] — another family of models without a normalised likelihood
- [[generative_models]] — the general concept of modelling how data arise

## References
- Goodfellow, I., et al. (2014). Generative adversarial nets. Advances in Neural Information Processing Systems, 27, 2672-2680.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
