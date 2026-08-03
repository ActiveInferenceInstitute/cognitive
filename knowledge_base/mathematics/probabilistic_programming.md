---
title: Probabilistic Programming
type: mathematical_concept
status: stable
created: 2024-03-20
tags:
  - mathematics
  - probabilistic_programming
  - inference
  - modeling
  - computation
semantic_relations:
  - type: implements
    links:
      - [[generative_models]]
      - [[bayesian_inference]]
      - [[variational_inference]]
  - type: related
    links:
      - [[graphical_models]]
      - [[factor_graphs]]
      - [[message_passing]]
  - type: applied_in
    links:
      - [[active_inference_theory]]
      - [[knowledge_base/cognitive/active_inference]]
---

# Probabilistic Programming

## Overview

Probabilistic programming is a paradigm that combines programming languages with probabilistic modeling, allowing users to specify generative models as programs and automatically perform inference. It provides a powerful abstraction for building and reasoning about [[generative_models|generative models]] without manually deriving inference algorithms.

In the context of [[active_inference_theory|active inference]], probabilistic programming frameworks enable rapid prototyping and deployment of complex [[generative_models|generative models]] for cognitive agents.

## Core Concepts

### Model Specification

Probabilistic programs define generative models through:

- **Random variables**: Stochastic primitives (distributions)
- **Deterministic computations**: Transformations of variables
- **Conditioning**: Constraining variables to observed values
- **Queries**: Computing posteriors over latent variables

### Inference Backends

Automatic inference algorithms include:

- **MCMC methods**: Hamiltonian Monte Carlo, NUTS, Metropolis-Hastings
- **Variational inference**: [[variational_inference|Automatic differentiation VI]]
- **[[message_passing|Message passing]]**: [[belief_propagation|Belief propagation]], expectation propagation
- **Importance sampling**: Sequential Monte Carlo, particle filtering

### Program Transformations

- **Automatic differentiation**: Gradient computation for optimization
- **Graph extraction**: Converting programs to [[graphical_models|graphical model]] representations
- **Compilation**: Optimizing model structure for efficient inference

## Key Frameworks

### Python Ecosystem

- **PyMC**: General-purpose Bayesian modeling with MCMC and VI
- **NumPyro/JAX**: High-performance probabilistic programming on JAX
- **Pyro**: Deep probabilistic programming on PyTorch
- **TensorFlow Probability**: Probabilistic layers and distributions
- **pymdp**: Active inference with discrete POMDP models

### Julia Ecosystem

- **RxInfer.jl**: Reactive [[message_passing|message passing]] for active inference
- **Turing.jl**: Universal probabilistic programming
- **Gen.jl**: Programmable inference

### Other Languages

- **Stan**: High-performance statistical modeling
- **WebPPL**: Probabilistic programming in the browser
- **Church/Venture**: Scheme-based probabilistic programming

## Connection to Active Inference

### Model-Based Agents

Probabilistic programs naturally express active inference agents:

```
# Pseudocode for active inference in a probabilistic program
prior = Normal(mu_prior, sigma_prior)      # Prior beliefs
state ~ prior                               # Sample hidden state
obs ~ Likelihood(state)                     # Generate observation
condition(obs == actual_observation)         # Condition on data
posterior = infer(state)                     # Compute posterior
```

### Automatic Inference for Agents

- Specify the [[generative_models|generative model]] declaratively
- Let the inference engine compute beliefs automatically
- Focus on model design rather than inference derivation

### RxInfer for Active Inference

The [[docs/implementation/rxinfer/README|RxInfer]] framework provides:

- Reactive message passing on [[factor_graphs|factor graphs]]
- Online inference for streaming data
- Native support for active inference models

## Advantages

- **Separation of concerns**: Model specification vs. inference algorithm
- **Rapid prototyping**: Quick iteration on model design
- **Composability**: Build complex models from simple components
- **Automatic inference**: No manual derivation of update equations
- **Reproducibility**: Models as executable specifications

## Applications

- [[knowledge_base/cognitive/active_inference|Active inference]] agent implementation
- [[bayesian_inference|Bayesian]] data analysis
- [[knowledge_base/cognitive/decision_making|Decision making]] under uncertainty
- Scientific modeling and simulation
- Machine learning model development

## Related Concepts

- [[generative_models]] - Models specified as probabilistic programs
- [[graphical_models]] - Graph-based model representations
- [[variational_inference]] - Approximate inference methods
- [[message_passing]] - Message-based inference
- [[factor_graphs]] - Graphical model framework
- [[bayesian_inference]] - Bayesian statistical methods

## References

- van de Meent, J.-W. et al. (2018). An Introduction to Probabilistic Programming
- Ghahramani, Z. (2015). Probabilistic machine learning and artificial intelligence
- Bagaev, D. et al. (2023). RxInfer: A Julia package for reactive real-time Bayesian inference
- Goodman, N. D. & Stuhlmuller, A. (2014). The Design and Implementation of Probabilistic Programming Languages

## See also

- [[knowledge_base/cognitive/probabilistic_programming|Probabilistic Programming]]
