---
title: Gibbs Distributions
description: Probability distributions proportional to the exponential of a negative energy, their maximum-entropy and free-energy variational characterisations, and their form as Gibbs random fields on graphs.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - statistical_physics
  - graphical_models
  - exponential_family
semantic_relations:
  - type: relates
    links:
      - "[[markov_random_fields]]"
      - "[[markov_properties]]"
      - "[[partition_function]]"
      - "[[maximum_entropy_principle]]"
      - "[[energy_based_models]]"
---

# Gibbs Distributions

## Overview
A Gibbs (or Boltzmann) distribution assigns to each configuration $x$ of a system with energy function $E(x)$ the probability
$$p(x)=\frac{1}{Z}\exp\!\left(-\frac{E(x)}{k_BT}\right),\qquad Z=\sum_x\exp\!\left(-\frac{E(x)}{k_BT}\right),$$
where $T$ is a temperature and $Z$ is the partition function. Low-energy configurations are exponentially favoured, and the temperature controls how sharply. The same form, with $k_BT$ absorbed into the energy, is used for any positive distribution, and it is the exponential family with the energy as sufficient statistic.

## Variational characterisations
Gibbs distributions arise in two equivalent ways. By the maximum entropy principle (Jaynes, 1957), $p$ maximises the Shannon entropy among distributions with a fixed mean energy, the Lagrange multiplier on that constraint being $1/k_BT$. Alternatively, for any distribution $q$ define the free energy functional $F[q]=\mathbb{E}_q[E]-k_BT\,H(q)$. Then
$$F[q]-F[p]=k_BT\,D_{\mathrm{KL}}(q\,\|\,p),\qquad F[p]=-k_BT\ln Z,$$
so $p$ is the unique minimiser of $F$. This identity is the origin of the variational free energy bound used in inference, where $E$ is the negative log of an unnormalised target density.

## Gibbs random fields
On a graph, let the energy decompose over cliques, $E(x)=\sum_C\phi_C(x_C)$, so that $p(x)\propto\prod_C\exp(-\phi_C(x_C))$. The Hammersley-Clifford theorem states that a strictly positive distribution is a Markov random field with respect to the graph if and only if it has this clique-factorised Gibbs form. Models of this kind include the Ising model, Boltzmann machines and image models used in Gibbs sampling (Geman and Geman, 1984).

## Relation to Active Inference
The link is a derivation at the level of the mathematics. Softmax policy and state posteriors are Gibbs distributions with expected free energy or negative log-joint as energy, and the variational free energy is the Gibbs free energy above. Steady-state densities in the free energy principle are written as $p\propto e^{-\text{surprise}}$ in the same way. Whether biological systems have physical temperatures matching these formal ones is a separate question.

## Related concepts
- [[partition_function]] — the normalising constant $Z$
- [[maximum_entropy_principle]] — the principle that yields this form
- [[markov_random_fields]] — graphical models whose joint is a Gibbs distribution
- [[energy_based_models]] — machine learning models defined through an energy
- [[gibbs_sampling]] — sampling that exploits conditional structure of such distributions
- [[exponential_family]] — the parametric class to which Gibbs distributions belong

## References
- Jaynes, E. T. (1957). Information theory and statistical mechanics. Physical Review, 106, 620-630.
- Geman, S., & Geman, D. (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. IEEE Transactions on Pattern Analysis and Machine Intelligence, 6, 721-741.
- Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models: Principles and Techniques. MIT Press.
