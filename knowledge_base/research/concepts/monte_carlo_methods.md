---
title: Monte Carlo Methods
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [monte_carlo, sampling, simulation, MCMC, particle_filter]
semantic_relations:
  - type: relates
    links: [[[sampling_strategies]], [[parameter_estimation]], [[knowledge_base/mathematics/stochastic_processes]], [[bayesian_analysis]]]
---

# Monte Carlo Methods

Monte Carlo methods for approximate inference and evaluation in Active Inference, including importance sampling, Markov chain Monte Carlo (MCMC), and particle filtering for online state estimation.

## Core Methods

### Simple Monte Carlo Estimation

```math
\mathbb{E}_p[f(x)] \approx \frac{1}{N} \sum_{i=1}^N f(x_i), \quad x_i \sim p(x)
```

Convergence rate: $O(1/\sqrt{N})$ regardless of dimensionality — the curse of dimensionality is largely absent.

### Importance Sampling

When sampling from $p$ is difficult, sample from a proposal $q$:

```math
\mathbb{E}_p[f(x)] = \mathbb{E}_q\left[f(x) \frac{p(x)}{q(x)}\right] \approx \sum_{i=1}^N \tilde{w}_i f(x_i) \quad \text{where } \tilde{w}_i = \frac{w_i}{\sum_j w_j}, \; w_i = \frac{p(x_i)}{q(x_i)}
```

### Effective Sample Size

```math
N_{\text{eff}} = \frac{1}{\sum_i \tilde{w}_i^2}
```

Low $N_{\text{eff}}$ indicates weight degeneracy — the proposal $q$ is a poor approximation of $p$.

## Particle Filtering for Active Inference

```python
class ParticleFilter:
    """Sequential Monte Carlo for online state estimation in Active Inference."""
    def __init__(self, n_particles, transition_fn, observation_fn):
        self.n_particles = n_particles
        self.transition = transition_fn
        self.observation = observation_fn
        self.particles = None
        self.weights = None

    def initialize(self, prior):
        self.particles = prior.sample(self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles

    def update(self, observation, action):
        # Propagate particles through transition model
        self.particles = self.transition(self.particles, action)
        # Compute observation likelihoods
        log_weights = self.observation(observation, self.particles)
        self.weights = np.exp(log_weights - np.max(log_weights))
        self.weights /= self.weights.sum()
        # Resample if effective sample size is low
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.n_particles / 2:
            self.resample()

    def resample(self):
        indices = np.random.choice(self.n_particles, size=self.n_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)
```

## MCMC Methods

### Metropolis-Hastings

```math
\alpha(x'|x) = \min\left(1, \frac{p(x')q(x|x')}{p(x)q(x'|x)}\right)
```

### Method Comparison

| Method | Online? | Multimodal? | Memory | Use Case |
| --- | --- | --- | --- | --- |
| Variational Bayes | No | Limited | O(d) | Standard Active Inference |
| Particle filter | Yes | Yes | O(N·d) | Online state estimation |
| MCMC | No | Yes | O(1) | Offline model fitting |
| Importance sampling | Batch | Yes | O(N) | Model evidence estimation |

## Related Topics

- [[sampling_strategies]] — Sampling design principles
- [[parameter_estimation]] — Parameter estimation methods
- [[bayesian_analysis]] — Bayesian inference framework
- [[knowledge_base/mathematics/stochastic_processes]] — Stochastic foundations\n