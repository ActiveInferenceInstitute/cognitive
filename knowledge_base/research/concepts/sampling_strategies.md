---
title: Sampling Strategies
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [sampling, statistics, experimental_design, inference]
semantic_relations:
  - type: relates
    links: [[[monte_carlo_methods]], [[experiment_design]], [[parameter_estimation]]]
---

# Sampling Strategies

Principled sampling strategies for Active Inference experiments, simulations, and approximate inference, covering both experimental sampling designs and computational sampling algorithms.

## Experimental Sampling

### Random Sampling
```python
def random_sample_conditions(parameter_space, n_samples):
    samples = {}
    for param, (low, high) in parameter_space.items():
        samples[param] = np.random.uniform(low, high, n_samples)
    return samples
```

### Latin Hypercube Sampling

Ensures better coverage of the parameter space:

```python
from scipy.stats import qmc

def latin_hypercube_sample(parameter_space, n_samples):
    sampler = qmc.LatinHypercube(d=len(parameter_space))
    unit_samples = sampler.random(n=n_samples)
    bounds = list(parameter_space.values())
    l_bounds = [b[0] for b in bounds]
    u_bounds = [b[1] for b in bounds]
    scaled = qmc.scale(unit_samples, l_bounds, u_bounds)
    return {name: scaled[:, i] for i, name in enumerate(parameter_space.keys())}
```

### Stratified Sampling

Ensures representation across categories:

```math
n_h = n \cdot \frac{N_h}{N}
```

where $n_h$ is the sample size for stratum $h$, $N_h$ is the stratum size, and $N$ is total population.

## Computational Sampling

| Method | Use Case | Convergence | Cost |
| --- | --- | --- | --- |
| Grid sampling | Low-dimensional | Exact coverage | $O(k^d)$ |
| Random | Any dimension | $O(1/\sqrt{N})$ | $O(N)$ |
| Latin hypercube | Medium dimension | Better coverage | $O(N)$ |
| Sobol sequences | High dimension | Quasi-random | $O(N)$ |
| MCMC | Posterior sampling | Asymptotic | Variable |

## Related Topics

- [[monte_carlo_methods]] — Monte Carlo techniques
- [[experiment_design]] — Experiment design
- [[parameter_estimation]] — Parameter estimation
