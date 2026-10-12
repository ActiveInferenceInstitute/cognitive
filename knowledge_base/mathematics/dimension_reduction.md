---
title: Dimension Reduction
description: Methods that represent high-dimensional data by a few informative variables, including principal component analysis, nonlinear embeddings and autoencoders, and their link to latent-variable and generative modelling.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistics
  - machine_learning
  - linear_algebra
semantic_relations:
  - type: relates
    links:
      - "[[latent_variable_models]]"
      - "[[variational_autoencoders]]"
      - "[[information_bottleneck]]"
      - "[[linear_algebra]]"
      - "[[decision_making]]"
---

# Dimension Reduction

## Overview
Dimension reduction maps data $x\in\mathbb{R}^D$ to a representation $z\in\mathbb{R}^d$ with $d\ll D$ while keeping the structure that matters for the task. It combats the curse of dimensionality, where distances and densities become unreliable as $D$ grows, and it supports visualisation, compression and simpler models. In decision settings the same idea appears as collapsing many attributes of an option into a few summary criteria.

## Linear methods
Principal component analysis (PCA) centres the data and diagonalises the sample covariance matrix $S=\frac1n X^\top X$. The first $d$ eigenvectors $u_1,\dots,u_d$ define the projection $z=U_d^\top x$. This choice maximises retained variance and, equivalently, minimises mean squared reconstruction error among all rank-$d$ linear projections. Computation is usually done through the singular value decomposition $X=U\Sigma V^\top$, and the Eckart-Young theorem states that truncating to the top $d$ singular values gives the best rank-$d$ approximation in Frobenius norm. Factor analysis adds a noise model, and probabilistic PCA is the corresponding latent-variable model with Gaussian latents and isotropic noise.

## Nonlinear methods
When data lie near a curved manifold, linear projection loses structure. Isomap preserves geodesic distances estimated on a neighbourhood graph, locally linear embedding preserves local reconstruction weights, and t-SNE matches neighbourhood probabilities for visualisation. Autoencoders learn an encoder-decoder pair by minimising reconstruction loss, and variational autoencoders add a probabilistic latent space regularised towards a prior.

## Relation to Active Inference
The connection is conceptual and in places technical. A generative model with a few hidden states producing high-dimensional observations is itself a dimension reduction, and perception is inference of the low-dimensional cause. The variational autoencoder objective is an evidence lower bound of the same family as variational free energy, so deep Active Inference implementations use it directly. Whether the brain implements anything like PCA is a separate empirical question and not asserted by the framework.

## Related concepts
- [[latent_variable_models]] — models where a few hidden variables generate many observations
- [[variational_autoencoders]] — learned nonlinear reduction trained with a free-energy-like bound
- [[information_bottleneck]] — compression that retains information relevant to a target
- [[feature_extraction]] — selecting or constructing informative inputs
- [[minimum_description_length]] — compression viewpoint on choosing representations

## References
- Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. Journal of Educational Psychology, 24, 417-441.
- Jolliffe, I. T. (2002). Principal Component Analysis (2nd ed.). Springer.
- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9, 2579-2605.
