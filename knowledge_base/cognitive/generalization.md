---
title: Generalization
description: Generalization is the extension of learned responses or knowledge from experienced cases to new ones, studied through generalization gradients, Shepard's law and error in statistical learning.
type: concept
status: stable
created: 2026-10-10
updated: 2026-10-10
tags:
  - learning
  - inference
  - model_selection
semantic_relations:
  - type: relates
    links:
      - "[[transfer_learning]]"
      - "[[model_complexity]]"
      - "[[structure_learning]]"
      - "[[statistical_learning]]"
aliases:
  - generalization_capacity
wikidata: Q6963735
---

# Generalization

## Overview
Generalization is the ability to respond appropriately to situations not encountered during learning. The same idea is studied in psychology, where a response learned to one stimulus is also elicited by similar stimuli, and in machine learning, where it denotes performance on unseen data. This page treats both senses, and notes that transfer between tasks is a related but distinct problem (see transfer_learning).

## Psychological and statistical accounts
**Generalization gradients.** After conditioning to one stimulus, responding to other stimuli falls off with their dissimilarity along some dimension, as shown classically in studies of tone frequency and colour. Shepard (1987) proposed a universal law: generalization probability decays approximately exponentially with distance in a suitably defined psychological space, $g(d) \approx e^{-d}$.

**Bayesian generalization.** Tenenbaum and Griffiths (2001) derived Shepard's law and related effects by treating generalization as inference about which hypothesis, or consequential region, the observed examples belong to. Generalization narrows as more examples accumulate, because hypotheses that are too broad become increasingly unlikely to have generated the data (the size principle).

**Machine learning.** Generalization error is the expected loss on new data from the same distribution. Expected squared error decomposes into squared bias, variance and irreducible noise: models that are too simple underfit and have high bias, while models that are too flexible overfit the training sample and have high variance. Regularisation, held-out validation and cross-validation are used to estimate and control it.

**Motor and perceptual learning.** Learning in one context often transfers only partially to others, with the pattern of transfer giving evidence about the underlying representation.

**Rule learning.** Humans can extract abstract rules from limited examples, such as in reasoning and problem solving, which supports claims of structured, compositional generalization.

## Role in Active Inference
Variational free energy equals the sum of an accuracy term and a complexity term that penalises divergence of the posterior from the prior. Complexity cost discourages parameters or structures that fit noise, so minimising free energy embodies an Occam factor, and Bayesian model selection by model evidence is a principled account of generalization. Structure learning, in which the model's form is chosen by evidence, extends this to the acquisition of abstractions. The connection is a derivation at the level of Bayesian model comparison; the empirical claim that brains approximate this trade-off is a hypothesis.

## Related concepts
- [[model_selection]] — comparing models by evidence and complexity
- [[perceptual_learning]] — specificity and transfer in perception
- [[motor_learning]] — transfer of skills across conditions
- [[reasoning_problem_solving]] — rule induction from examples
- [[neural_plasticity]] — the basis for learned changes that are generalised

## References
- Shepard, R. N. (1987). Toward a universal law of generalization for psychological science. Science.
- Tenenbaum, J. B., & Griffiths, T. L. (2001). Generalization, similarity, and Bayesian inference. Behavioral and Brain Sciences.
