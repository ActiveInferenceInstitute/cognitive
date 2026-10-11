---
title: Neural Sampling
description: The hypothesis that variable neural activity represents samples from a probability distribution over latent causes, its predictions, contrast with population-code accounts, and its relation to Active Inference.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - bayesian_brain
  - sampling
  - neural_coding
semantic_relations:
  - type: relates
    links:
      - "[[bayesian_brain]]"
      - "[[markov_chain_monte_carlo]]"
      - "[[predictive_coding]]"
      - "[[neural_coding]]"
      - "[[population_coding]]"
      - "[[langevin_dynamics]]"
---

# Neural Sampling

## Overview
Neural sampling is a proposal about how neural circuits could represent and compute with uncertainty. At any moment, activity such as membrane potentials or spikes is taken to be a sample from the posterior distribution over the causes of sensory input, and the stochastic fluctuations of activity over time correspond to drawing successive samples. Uncertainty is then carried by the variability of the samples, and expectations of any function of the latent variables are obtained by averaging over time.

## Computational basis
A recurrent network with stochastic dynamics can implement Markov chain Monte Carlo. Networks of stochastic spiking neurons with suitable connectivity have a stationary distribution over binary activity patterns, and continuous-valued variants correspond to Langevin dynamics, which descend the gradient of a log posterior while injecting noise. Samples are correlated in time, so accurate estimates need multiple steps, which makes speed a central question for this account.

## Evidence and contrast
Predicted signatures include trial-to-trial variability that depends on stimulus uncertainty, and spontaneous activity that resembles the average of stimulus-evoked activity, as the prior over images should when no stimulus is present. Berkes and colleagues reported such a similarity in the visual cortex of developing ferrets. The main alternative is the probabilistic population code, in which a population's instantaneous firing pattern parameterises a distribution, as in the framework of Ma and colleagues. The two accounts make different predictions, and the evidence does not yet decide between them. The proposals may also be combined.

## Relation to Active Inference
Active Inference in its usual form assumes that neuronal activity encodes the parameters of an approximate posterior, such as sufficient statistics under a Gaussian assumption, and not samples. Message passing minimises free energy deterministically, with noise treated as a nuisance. Neural sampling is therefore a competing or complementary hypothesis about the same problem, belief representation in the brain. Both accounts belong to the wider Bayesian brain hypothesis, and which of them best describes cortex is an open empirical question.

## Related concepts
- [[bayesian_brain]] — the hypothesis that neural computation implements inference
- [[markov_chain_monte_carlo]] — the algorithmic template for sampling dynamics
- [[predictive_coding]] — a parametric alternative for representing beliefs
- [[population_coding]] — distributions encoded across a population's activity
- [[langevin_dynamics]] — continuous noisy dynamics that sample from a density

## References
- Ma, W. J., Beck, J. M., Latham, P. E., & Pouget, A. (2006). Bayesian inference with probabilistic population codes. Nature Neuroscience, 9, 1432-1438.
- Fiser, J., Berkes, P., Orbán, G., & Lengyel, M. (2010). Statistically optimal perception and learning: from behavior to neural representations. Trends in Cognitive Sciences, 14, 119-130.
- Berkes, P., Orbán, G., Lengyel, M., & Fiser, J. (2011). Spontaneous cortical activity reveals hallmarks of an optimal internal model of the environment. Science, 331, 83-87.
- Buesing, L., Bill, J., Nessler, B., & Maass, W. (2011). Neural dynamics as sampling: a model for stochastic computation in recurrent networks of spiking neurons. PLoS Computational Biology, 7, e1002211.
