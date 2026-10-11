---
title: Gibbs Sampling
description: A Markov chain Monte Carlo algorithm that samples a joint distribution by repeatedly drawing each variable from its conditional distribution given the rest, widely used with graphical models.
type: mathematical_concept
status: stable
created: 2026-10-11
updated: 2026-10-11
tags:
  - sampling
  - mcmc
  - inference
semantic_relations:
  - type: relates
    links:
      - "[[markov_chain_monte_carlo]]"
      - "[[markov_random_fields]]"
      - "[[metropolis_hastings]]"
      - "[[conditional_probability]]"
      - "[[variational_inference]]"
---

# Gibbs Sampling

## Overview
Gibbs sampling is a Markov chain Monte Carlo method for drawing samples from a joint distribution $p(x_1,\dots,x_n)$ when direct sampling is hard but each full conditional $p(x_i \mid x_{-i})$ is easy to sample. It produces dependent samples whose long-run distribution is the target, and these are used to estimate expectations and marginals.

## Algorithm
Starting from an initial state $x^{(0)}$, one sweep updates the variables in turn:
$$x_i^{(t+1)} \sim p\big(x_i \mid x_1^{(t+1)},\dots,x_{i-1}^{(t+1)},x_{i+1}^{(t)},\dots,x_n^{(t)}\big).$$
Variables may be visited in a fixed or random order. The joint distribution is invariant under each update, so the chain has the target as a stationary distribution, and under mild conditions such as positivity it is ergodic. Gibbs sampling is a special case of the Metropolis-Hastings algorithm in which every proposal is accepted.

## Use with graphical models
In a Markov random field or Bayesian network the conditional of one variable depends only on its Markov blanket: its neighbours in an undirected graph, or its parents, children and co-parents in a directed one. Each update is therefore local and cheap, and can often be parallelised across variables that are not neighbours. Variants include blocked sampling, which updates groups of variables jointly, and collapsed sampling, which integrates out some variables analytically.

## Limitations
Convergence can be slow when variables are strongly correlated, because single-site moves explore the space in small steps. Samples are autocorrelated, so the effective sample size is lower than the chain length, and a burn-in period is usually discarded.

## Relation to Active Inference
Active Inference usually relies on variational methods, which turn inference into optimisation of a free energy, instead of sampling. Mean-field variational updates have a structure similar to Gibbs sweeps, since each factor is updated given expectations over the others, but they update distributions instead of drawing samples. Sampling-based accounts of neural computation, in which variability in neural activity represents draws from a posterior, are a separate hypothesis and are not a core commitment of Active Inference.

## Related concepts
- [[markov_chain_monte_carlo]] — the family of methods to which Gibbs sampling belongs
- [[metropolis_hastings]] — the general algorithm of which Gibbs sampling is a special case
- [[markov_random_fields]] — a model class with simple local conditionals
- [[monte_carlo_methods]] — the general sampling framework
- [[neural_sampling]] — the hypothesis that neural variability implements sampling
- [[variational_inference]] — the optimisation-based alternative

## References
- Geman, S., & Geman, D. (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. IEEE Transactions on Pattern Analysis and Machine Intelligence, 6, 721-741.
- Gelfand, A. E., & Smith, A. F. M. (1990). Sampling-based approaches to calculating marginal densities. Journal of the American Statistical Association, 85, 398-409.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
