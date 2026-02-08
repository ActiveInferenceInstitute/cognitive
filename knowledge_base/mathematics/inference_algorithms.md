---
title: Inference Algorithms
type: mathematical_concept
status: stable
created: 2024-03-20
tags:
  - mathematics
  - inference
  - algorithms
  - computation
  - probabilistic
semantic_relations:
  - type: includes
    links:
      - [[belief_propagation]]
      - [[variational_inference]]
      - [[message_passing]]
  - type: operates_on
    links:
      - [[graphical_models]]
      - [[factor_graphs]]
      - [[generative_models]]
  - type: foundation_for
    links:
      - [[active_inference_theory]]
      - [[free_energy_principle]]
---

# Inference Algorithms

## Overview

Inference algorithms are computational methods for computing posterior distributions, marginal probabilities, or maximum a posteriori (MAP) estimates in probabilistic models. They are the computational engine behind [[active_inference_theory|active inference]], enabling agents to update beliefs about hidden states given observations.

## Classes of Inference

### Exact Inference

Methods that compute exact posterior distributions:

- **Variable elimination**: Sequentially marginalize variables
- **Junction tree algorithm**: Transform to clique tree for efficient computation
- **[[belief_propagation|Belief propagation]] on trees**: Exact marginals via message passing
- **Matrix methods**: Direct computation for small discrete models

### Approximate Inference

#### Variational Methods

[[variational_inference|Variational inference]] reformulates inference as optimization:

$$q^*(s) = \arg\min_{q \in \mathcal{Q}} D_{KL}[q(s) \| p(s | o)]$$

- **Mean-field approximation**: Factorized posterior
- **Structured variational inference**: Structured approximating families
- **Amortized inference**: Neural network encoders

#### Sampling Methods

Monte Carlo methods for posterior approximation:

- **Markov Chain Monte Carlo (MCMC)**: Metropolis-Hastings, Gibbs sampling
- **Hamiltonian Monte Carlo**: Gradient-based exploration
- **Sequential Monte Carlo**: Particle filtering for temporal models
- **Importance sampling**: Weighted samples from proposal distribution

#### Message Passing

[[message_passing|Message passing]] algorithms on [[graphical_models|graphical models]]:

- **[[belief_propagation]]**: Sum-product and max-product algorithms
- **Expectation propagation**: Moment matching approximations
- **Variational message passing**: Natural parameter updates
- **Constrained Bethe free energy**: Generalized message passing

### Gradient-Based Inference

- **Natural gradient descent**: Information-geometric optimization
- **Laplace approximation**: Gaussian approximation at the mode
- **Expectation-maximization (EM)**: Iterative parameter estimation

## Inference in Active Inference

### Perception as Inference

In [[knowledge_base/cognitive/active_inference|active inference]], perception is state estimation:

$$q(s_t) = \arg\min_q F[q, o_t]$$

where $F$ is the [[variational_free_energy|variational free energy]].

### Planning as Inference

Policy evaluation through expected free energy:

$$G(\pi) = \sum_\tau G(\pi, \tau)$$

using tree search or sampling over policy space.

### Learning as Inference

Parameter updates as Bayesian learning:

$$q(\theta) \propto p(o | \theta) \, p(\theta)$$

## Computational Considerations

### Complexity

- Exact inference is NP-hard in general [[graphical_models|graphical models]]
- Approximate methods trade accuracy for tractability
- [[factor_graphs|Factor graph]] structure determines efficient algorithms

### Convergence

- Variational methods converge to local optima
- MCMC methods converge to true posterior asymptotically
- [[belief_propagation|Loopy BP]] may not converge on all graphs

### Scalability

- Amortized inference scales to large datasets
- Distributed message passing for parallel computation
- GPU acceleration for variational and sampling methods

## Related Concepts

- [[belief_propagation]] - Message passing inference
- [[variational_inference]] - Variational methods
- [[message_passing]] - Message passing framework
- [[graphical_models]] - Model representations
- [[factor_graphs]] - Factor graph structures
- [[generative_models]] - Models to perform inference on
- [[probabilistic_programming]] - Automated inference

## References

- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational Inference: A Review for Statisticians
- Koller, D. & Friedman, N. (2009). Probabilistic Graphical Models, Ch. 9-13
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective
- Parr, T. & Friston, K. J. (2019). Generalised free energy and active inference
