---
title: Cross Entropy
description: The expected code length when events from distribution p are encoded with a code built for q, its decomposition into entropy plus KL divergence, and its appearance in free energy and learning objectives.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - information_theory
  - loss_functions
  - kl_divergence
semantic_relations:
  - type: relates
    links:
      - "[[entropy]]"
      - "[[kl_divergence]]"
      - "[[maximum_likelihood]]"
      - "[[information_theory]]"
      - "[[variational_free_energy]]"
wikidata: Q1685498
---

# Cross Entropy

## Overview
For two distributions $p$ and $q$ over the same discrete space, the cross entropy of $q$ relative to $p$ is
$$
H(p,q)=-\sum_x p(x)\log q(x)=H(p)+D_{\mathrm{KL}}(p\,\|\,q).
$$
It is the average number of nats (or bits) needed to encode samples drawn from $p$ using a code optimal for $q$. Since $D_{\mathrm{KL}}\ge0$, $H(p,q)\ge H(p)$, with equality only when $q=p$. The measure is not symmetric: $H(p,q)\neq H(q,p)$ in general. Not to be confused with joint entropy, which has a similar notation $H(X,Y)$ but concerns two variables rather than two distributions.

## Use as a loss function
Because $H(p)$ does not depend on $q$, minimising $H(p,q)$ over $q$ is the same as minimising $D_{\mathrm{KL}}(p\,\|\,q)$. When $p$ is the empirical distribution of a dataset, $H(p,q_\theta)$ equals the average negative log-likelihood, so minimising cross entropy is maximum likelihood estimation. This is the origin of the cross-entropy loss used to train classifiers, where $p$ is a one-hot label and $q_\theta$ is a softmax output, giving the loss $-\log q_\theta(y\mid x)$. The same quantity, expressed in bits, is the average code length that an entropy coder achieves when the model $q$ is used for compression.

## Relation to Active Inference
The link is a direct identity. Variational free energy for observations $o$ and hidden states $s$ with approximate posterior $q(s)$ can be written
$$
F=\mathbb{E}_{q}[-\ln p(o,s)]-H[q(s)],
$$
the cross entropy between $q$ and the generative model's joint density, minus the entropy of $q$. Free energy minimisation therefore asks for a belief that is both compatible with the model and as uncertain as the data allow. Expected free energy has related decompositions, and in prediction-error formulations of perception the cross-entropy term reduces to precision-weighted squared errors under Gaussian assumptions. Using cross entropy as a training loss for the likelihood mapping is standard in deep-learning implementations.

## Related concepts
- [[entropy]] — the term $H(p)$ that cross entropy exceeds
- [[kl_divergence]] — the excess code length over the entropy
- [[maximum_likelihood]] — equivalent to minimising empirical cross entropy
- [[information_theory]] — the framework in which code length is defined
- [[variational_free_energy]] — contains a cross-entropy term

## References
- Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
