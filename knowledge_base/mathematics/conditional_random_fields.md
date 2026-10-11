---
title: Conditional Random Fields
description: Discriminative undirected graphical models that define a conditional distribution over structured outputs given observations, their linear-chain form, training and inference, and contrast with generative models used in Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - graphical_models
  - structured_prediction
  - discriminative_models
semantic_relations:
  - type: relates
    links:
      - "[[markov_random_fields]]"
      - "[[factor_graphs]]"
      - "[[graphical_models]]"
      - "[[conditional_independence]]"
      - "[[message_passing]]"
---

# Conditional Random Fields

## Overview
A conditional random field (CRF) is an undirected probabilistic graphical model of the conditional distribution $p(\mathbf{y}\mid\mathbf{x})$ of a set of output variables $\mathbf{y}$, typically labels with dependencies among them, given observed inputs $\mathbf{x}$. It was introduced by Lafferty, McCallum and Pereira (2001) for segmenting and labelling sequences. Because it models the conditional directly, it does not need to model, or make independence assumptions about, the distribution of the inputs.

## Definition
Over a graph $G$ with cliques $c$, the model is

$$p(\mathbf{y}\mid\mathbf{x})=\frac{1}{Z(\mathbf{x})}\prod_{c}\psi_c(\mathbf{y}_c,\mathbf{x}),\qquad Z(\mathbf{x})=\sum_{\mathbf{y}}\prod_{c}\psi_c(\mathbf{y}_c,\mathbf{x}),$$

where each potential is usually log-linear, $\psi_c=\exp\big(\mathbf{w}^{\top}\mathbf{f}_c(\mathbf{y}_c,\mathbf{x})\big)$, with feature functions $\mathbf{f}_c$ and weights $\mathbf{w}$. Conditional on $\mathbf{x}$ the labels form a Markov random field. The normaliser $Z(\mathbf{x})$ depends on the input, which distinguishes a CRF from a Markov random field over $(\mathbf{x},\mathbf{y})$. The best-known case is the **linear-chain CRF** for sequence labelling, in which $\mathbf{y}$ forms a chain and features may depend on the entire input sequence.

## Inference and training
For chains and trees, $Z(\mathbf{x})$, marginals and the most probable labelling are computed exactly by dynamic programming of the forward-backward and Viterbi type, which are instances of [[message_passing]] on a factor graph. For loopy graphs, exact inference is generally intractable, and approximations such as loopy belief propagation or variational methods are used. Weights are fit by maximising the conditional log-likelihood, which is concave for log-linear potentials, using gradient methods. The gradient is the difference between empirical and model-expected feature counts, so training needs the marginals from inference.

## Relation to Active Inference
The link is an analogy through shared graphical-model machinery. Active Inference models are generative: they specify a joint distribution over hidden states, observations and policies, and infer beliefs by minimising free energy through message passing on factor graphs. A CRF is discriminative: it learns a mapping from inputs to structured outputs without a model of how the inputs arise, so it has no counterpart to generating predictions or to epistemic action. Both rely on the same structural idea, a factorised distribution whose conditional independencies make inference tractable. CRFs are not used as standard components of Active Inference agents.

## Related concepts
- [[markov_random_fields]] — the undirected model that CRFs condition on inputs
- [[factor_graphs]] — the representation used to run inference
- [[graphical_models]] — the wider family
- [[conditional_independence]] — the structure encoded by the graph
- [[message_passing]] — the inference algorithms for chains and trees

## References
- Lafferty, J., McCallum, A., & Pereira, F. (2001). Conditional random fields: probabilistic models for segmenting and labeling sequence data. Proceedings of the 18th International Conference on Machine Learning, 282-289.
- Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models: Principles and Techniques. MIT Press.
