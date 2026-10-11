---
title: Energy-Based Models
description: Models that define a probability distribution or compatibility score through an energy function, including Boltzmann machines and Hopfield networks, with training difficulties and links to free energy.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - machine_learning
  - statistical_physics
  - boltzmann_distribution
semantic_relations:
  - type: relates
    links:
      - "[[partition_function]]"
      - "[[hopfield_networks]]"
      - "[[statistical_physics]]"
      - "[[free_energy]]"
      - "[[predictive_coding]]"
---

# Energy-Based Models

## Overview
An energy-based model (EBM) assigns a scalar energy $E_\theta(x)$ to each configuration $x$, with low energy for configurations that are compatible with the data and high energy for those that are not. When a normalised probability is required, the energy defines a Boltzmann (Gibbs) distribution,
$$
p_\theta(x)=\frac{e^{-E_\theta(x)}}{Z_\theta},\qquad Z_\theta=\sum_x e^{-E_\theta(x)}\ \text{or}\ \int e^{-E_\theta(x)}dx.
$$
The normaliser $Z_\theta$ is the partition function. LeCun and colleagues (2007) argued for a view in which learning shapes the energy surface and need not involve explicit normalisation.

## Examples and training
Classical EBMs include Hopfield networks, Boltzmann machines (Ackley et al., 1985) and restricted Boltzmann machines. Modern variants parameterise $E_\theta$ with deep networks. The log-likelihood gradient has the form
$$
\nabla_\theta\ln p_\theta(x)=-\nabla_\theta E_\theta(x)+\mathbb{E}_{x'\sim p_\theta}\big[\nabla_\theta E_\theta(x')\big],
$$
a data term that lowers energy at observed points and a model term that raises it elsewhere. The model expectation requires sampling, usually with Markov chain Monte Carlo, which is the main computational difficulty. Contrastive divergence (Hinton, 2002) approximates it with a few sampling steps started from the data. Alternatives avoid $Z_\theta$ through score matching, noise-contrastive estimation or adversarial training. Sampled chains may mix poorly in high dimensions, and the learned energy can assign low energy to regions far from the data.

## Relation to Active Inference
The relationship is formal and partly terminological. Variational free energy has the same structure as the free energy of statistical physics, expected energy minus entropy, with the "energy" equal to $-\ln p(o,s)$ under the agent's generative model. Predictive coding networks minimise a sum of squared prediction errors that can be read as an energy function, and Hopfield-like attractor networks give a related picture of memory retrieval. These are analogies of mathematical form: a generic EBM is learned from data by gradient steps, whereas Active Inference starts from an explicit generative model and minimises free energy over beliefs.

## Related concepts
- [[partition_function]] — the normalising constant that makes training hard
- [[hopfield_networks]] — an early energy-based associative memory
- [[statistical_physics]] — the source of the Boltzmann form
- [[free_energy]] — energy minus entropy, in physics and in inference
- [[predictive_coding]] — error-minimising networks with an energy interpretation

## References
- Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. (1985). A learning algorithm for Boltzmann machines. Cognitive Science, 9, 147-169.
- Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. Neural Computation, 14, 1771-1800.
- LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. J. (2007). A tutorial on energy-based learning. In G. Bakir et al. (Eds.), Predicting Structured Data (pp. 191-246). MIT Press.
