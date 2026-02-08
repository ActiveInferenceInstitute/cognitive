---
title: Generative Models
type: mathematical_concept
status: stable
created: 2024-03-20
tags:
  - mathematics
  - generative_models
  - probability
  - active_inference
  - bayesian
semantic_relations:
  - type: foundation_for
    links:
      - [[active_inference_theory]]
      - [[free_energy_principle]]
      - [[belief_updating]]
  - type: uses
    links:
      - [[graphical_models]]
      - [[probability_theory]]
      - [[bayesian_inference]]
  - type: related
    links:
      - [[probabilistic_models]]
      - [[variational_inference]]
      - [[message_passing]]
---

# Generative Models

## Overview

A generative model is a probabilistic model that specifies a joint distribution over observable variables and latent (hidden) variables. It describes the causal process by which observations are generated, enabling inference about hidden causes from observed effects.

In [[active_inference_theory|active inference]], the generative model is the agent's internal model of how sensory observations arise from hidden states of the world. Perception corresponds to inverting this model via [[belief_propagation|belief propagation]] or [[variational_inference|variational inference]].

## Mathematical Definition

### Joint Distribution

A generative model specifies the joint distribution:

$$p(o, s, \theta) = p(o | s, \theta) \, p(s | \theta) \, p(\theta)$$

where:
- $o$ = observations (sensory data)
- $s$ = hidden states (latent variables)
- $\theta$ = model parameters

### Likelihood and Prior

The model factorizes into:

- **Likelihood** $p(o | s)$: How hidden states generate observations
- **Prior** $p(s)$: Prior beliefs about hidden states
- **Hyperpriors** $p(\theta)$: Beliefs about model parameters

### Model Inversion

Inference corresponds to computing the posterior via Bayes' rule:

$$p(s | o) = \frac{p(o | s) \, p(s)}{p(o)}$$

Since the evidence $p(o) = \int p(o, s) \, ds$ is often intractable, approximate methods like [[variational_inference|variational inference]] are used.

## Generative Models in Active Inference

### POMDP Generative Models

In [[active_inference_pomdp|discrete active inference]], the generative model includes:

- **A matrix** (likelihood): $p(o_t | s_t)$
- **B matrix** (transitions): $p(s_{t+1} | s_t, a_t)$
- **C vector** (preferences): $p(o)$ encoding preferred observations
- **D vector** (initial state): $p(s_0)$

### Continuous Generative Models

For continuous state spaces:

$$\dot{s} = f(s, a) + \omega, \quad o = g(s) + \nu$$

where $f$ is the flow function, $g$ is the observation function, and $\omega, \nu$ are noise.

### Hierarchical Generative Models

Multi-level models capture structure at different scales:

$$p(o, s^{(1)}, \ldots, s^{(L)}) = p(o | s^{(1)}) \prod_{l=1}^{L-1} p(s^{(l)} | s^{(l+1)}) \, p(s^{(L)})$$

This underlies [[knowledge_base/cognitive/hierarchical_processing|hierarchical predictive processing]].

### Deep Temporal Models

For sequential data with temporal depth:

$$p(\tilde{o}, \tilde{s} | \pi) = p(s_0) \prod_{\tau=0}^{T} p(o_\tau | s_\tau) \, p(s_{\tau+1} | s_\tau, \pi)$$

where $\pi$ denotes the policy.

## Connection to Free Energy

The [[free_energy_principle|free energy principle]] relates generative models to inference:

### Variational Free Energy

$$F = D_{KL}[q(s) \| p(s | o)] - \ln p(o)$$

Minimizing $F$ with respect to $q(s)$ performs approximate Bayesian inference.

### Expected Free Energy

For action selection, the expected free energy under policy $\pi$:

$$G(\pi) = \sum_\tau \mathbb{E}_{q(o_\tau, s_\tau | \pi)} [\ln q(s_\tau | \pi) - \ln p(o_\tau, s_\tau)]$$

balances [[knowledge_base/mathematics/epistemic_value|epistemic value]] (information gain) and pragmatic value (preference satisfaction).

## Model Classes

### Parametric Models

- Linear Gaussian models
- Exponential family models
- Neural network parameterizations

### Non-Parametric Models

- Gaussian processes
- Dirichlet processes
- Indian buffet processes

### Structured Models

- [[graphical_models|Graphical models]] (directed, undirected, factor graphs)
- State-space models
- Mixture models

## Learning

### Parameter Learning

Update model parameters to better explain data:

$$\theta^* = \arg\max_\theta \, \mathbb{E}_{q(s)}[\ln p(o, s | \theta)]$$

### Structure Learning

Discover the graph structure of the generative model:

- Model comparison via free energy / model evidence
- Bayesian model selection
- [[bayesian_inference|Bayesian model averaging]]

## Applications

- [[knowledge_base/cognitive/active_inference|Active inference]] agents
- [[knowledge_base/cognitive/predictive_processing|Predictive processing]] in the brain
- Perceptual inference
- [[knowledge_base/cognitive/decision_making|Decision making]]
- [[probabilistic_programming|Probabilistic programming]] implementations

## Related Concepts

- [[graphical_models]] - Graph-based probabilistic models
- [[probabilistic_models]] - General probabilistic modeling
- [[variational_inference]] - Approximate inference methods
- [[belief_propagation]] - Message passing inference
- [[active_inference_theory]] - Active inference framework
- [[free_energy_principle]] - Unifying theoretical principle
- [[bayesian_inference]] - Bayesian methods

## References

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning
- Goodfellow, I., et al. (2014). Generative Adversarial Nets
