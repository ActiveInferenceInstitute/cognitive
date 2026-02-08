---
title: Probabilistic Programming
type: concept
status: stable
created: 2026-02-06
updated: 2026-02-07
tags:
  - probabilistic-programming
  - bayesian-inference
  - generative-models
  - variational-inference
  - implementation
semantic_relations:
  - type: relates
    links:
      - [[bayesian_inference]]
      - [[generative_model]]
      - [[knowledge_base/mathematics/probability_distributions]]
      - [[computational_neuroscience]]
---

# Probabilistic Programming

## Overview

Probabilistic programming languages (PPLs) provide a natural implementation substrate for Active Inference by allowing practitioners to specify generative models as programs and automatically derive inference algorithms. The generative model is the program; perception is program inversion.

## Key Frameworks

| Framework | Language | Inference Methods | Active Inference Use |
| --- | --- | --- | --- |
| PyMC | Python | NUTS, VI, SMC | Model fitting, parameter estimation |
| Stan | Stan/R/Python | HMC, ADVI | Bayesian model comparison |
| NumPyro | Python/JAX | NUTS, SVI | GPU-accelerated inference |
| Pyro | Python/PyTorch | SVI, deep VI | Deep Active Inference |
| WebPPL | JavaScript | Enumeration, MCMC | Cognitive modeling |
| Gen | Julia | Custom MH, PF | Flexible Active Inference |

## Active Inference as Probabilistic Program

### Generative Model

```python
import pymc as pm
import numpy as np

def active_inference_model(observations, n_states, n_obs):
    with pm.Model() as model:
        # Prior over initial states (D)
        D = pm.Dirichlet('D', a=np.ones(n_states))
        
        # Observation model (A)
        A = pm.Dirichlet('A', a=np.ones(n_obs), shape=(n_states, n_obs))
        
        # Initial state
        s = pm.Categorical('s0', p=D)
        
        # Generate observations
        for t, obs in enumerate(observations):
            o = pm.Categorical(f'o_{t}', p=A[s], observed=obs)
    
    return model
```

### Automated Inference

The power of PPLs is that inference is automatic:
1. Specify generative model
2. Condition on observations
3. PPL infers posterior via MCMC, variational inference, or exact enumeration

### Connection to Message Passing

PPL inference algorithms correspond to Active Inference update rules:
- **Variational inference**: Minimizes $F = D_{KL}[q||p] - \ln p(o)$
- **MCMC sampling**: Explores posterior via Markov chains
- **Particle filtering**: Sequential Monte Carlo for online inference

## Advantages for Active Inference Research

1. **Rapid prototyping**: Specify new generative models in minutes
2. **Automatic gradients**: PPLs handle differentiation for variational updates
3. **Model comparison**: Built-in tools for computing model evidence
4. **Modularity**: Composable model components

## Related Topics

- [[bayesian_inference]] — Bayesian inference theory
- [[generative_model]] — Generative models
- [[knowledge_base/mathematics/probability_distributions]] — Probability distributions
- [[computational_neuroscience]] — Computational approaches
- [[active_inference]] — Active Inference framework
