---
title: Unsupervised Learning
type: concept
status: stable
created: 2026-10-09
updated: 2026-10-09
tags:
  - learning
  - machine_learning
  - representation
semantic_relations:
  - type: relates
    links:
      - "[[active_inference]]"
      - "[[predictive_coding]]"
      - "[[generative_model]]"
wikidata: Q1152135
---

# Unsupervised Learning

## Overview
Unsupervised learning is learning structure from data without target labels or explicit reward: the learner must find regularities, clusters, latent factors, or compact codes in the inputs themselves. In machine learning it contrasts with supervised learning (labeled targets) and reinforcement learning (scalar reward). In cognitive science it names the learning, such as category formation and perceptual tuning, that happens through exposure alone.

## Mechanism
Typical objectives fall into a few families:

- **Density estimation / generative modeling**: fit $p(x)$, for example with mixture models, variational autoencoders, or normalizing flows.
- **Dimensionality reduction and latent factors**: principal component analysis, independent component analysis, and factor analysis find lower-dimensional descriptions.
- **Clustering**: partition inputs by similarity.
- **Sparse coding**: represent each input with few active units from an overcomplete dictionary. Olshausen and Field (1996) showed that sparse coding of natural images yields receptive fields resembling simple cells in primary visual cortex.
- **Self-supervised prediction**: predict one part of the input from another, using the data itself as the target.

Biological candidates include Hebbian-type rules with normalization or competition, which can extract principal components or perform winner-take-all clustering (see [[hebbian_learning]]), and statistical learning in infants and adults, where transition regularities are acquired without instruction (see [[statistical_learning]]).

## Role in Active Inference
Learning in active inference is unsupervised with respect to labels. The agent optimizes a generative model by minimizing variational free energy, which is a bound on the log evidence of its own observations; it needs no teacher. This makes perception and learning a form of generative modeling, and [[predictive_coding]] networks are one concrete implementation, learning hierarchical latent causes by minimizing prediction error at each level. The distinction from classical unsupervised methods is that an active inference agent also selects its data through action, so its learning is shaped by [[epistemic_value]] and by prior preferences, not by a fixed dataset. The agent is therefore not purely unsupervised in practice: preferences and the consequences of action provide a task-relevant signal. Whether fitting $p(x)$ yields representations that are useful for behavior is an empirical question that depends on the model class, not something the free-energy objective guarantees.

## Related concepts
- [[predictive_coding]] — unsupervised hierarchical learning by error minimization
- [[generative_model]] — the object being fitted without labels
- [[statistical_learning]] — human and animal unsupervised extraction of regularities
- [[pattern_recognition]] — categories discovered from structure
- [[structure_learning]] — discovering model form without supervision
- [[reinforcement_learning]] — the paradigm that adds reward to the picture

## References
- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*.
- Oja, E. (1982). A simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology*.
