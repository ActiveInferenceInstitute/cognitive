---
title: Probability Distributions
type: mathematical_concept
status: stable
created: 2025-03-03
updated: 2026-10-09
tags:
- probability
- mathematics
aliases:
  - categorical_distribution
semantic_relations:
  - type: relates_to
    links:
      - "[[entropy]]"
      - "[[kl_divergence]]"
      - "[[exponential_family]]"
---


# Probability Distributions

Active Inference models are written in terms of a few families of distributions. Discrete models use categorical distributions for states, observations and policies, with Dirichlet distributions as priors over the categorical parameters. Continuous models use Gaussian distributions, where precision (inverse variance) is the central quantity. All three families belong to the exponential family, which is why their entropies, KL divergences and conjugate updates have closed forms.

## Categorical

A categorical distribution over $K$ outcomes is a vector $\pi$ with $\pi_k \ge 0$ and $\sum_k \pi_k = 1$. Its entropy is $H[\pi] = -\sum_k \pi_k \ln \pi_k$, maximal at $\ln K$ for the uniform vector. Examples: $H(0.5, 0.5) = 0.6931$ nats and $H(0.9, 0.1) = 0.3251$ nats.

In the repository (`code/tools/src/models/active_inference/generative_model.py`) every discrete object is categorical:

- each column of `A` is $P(o \mid s)$ for one state, and `DiscreteGenerativeModel` rejects matrices whose columns do not sum to one;
- each slice `B[:, :, a]` has categorical columns $P(s' \mid s, a)$;
- `D` is the prior over states, `E` the prior over actions, and `preferences` is $\mathrm{softmax}(C)$, a categorical distribution over observations.

Policy probabilities are categorical too: $P(\pi) = \mathrm{softmax}(-G/T)$ (see [[softmax_function]] and [[temperature_parameter]]).

## Dirichlet

The Dirichlet distribution with concentration $\alpha \in \mathbb{R}_{>0}^K$ is a distribution over categorical parameters, $p(\pi) \propto \prod_k \pi_k^{\alpha_k - 1}$. It is conjugate to the categorical likelihood: after observing counts $n_k$, the posterior is Dirichlet with $\alpha_k + n_k$. This is how likelihood and transition matrices are learned in discrete Active Inference, with $\alpha$ acting as pseudo-counts. Useful moments are $\mathbb{E}[\pi_k] = \alpha_k / \alpha_0$ and $\mathbb{E}[\ln \pi_k] = \psi(\alpha_k) - \psi(\alpha_0)$, with $\alpha_0 = \sum_k \alpha_k$ and $\psi$ the digamma function. For $\alpha = (2, 3)$ these give $\mathbb{E}[\ln \pi] = (-13/12,\ -7/12) = (-1.0833,\ -0.5833)$; a Monte Carlo average over 400000 draws (seed 0) gives $(-1.083,\ -0.583)$.

The repository's `sampling` belief update draws particles from `rng.dirichlet(posterior * concentration)` and averages them, using the Dirichlet as a distribution over beliefs that is centred on the exact posterior.

## Gaussian

A univariate Gaussian $\mathcal{N}(\mu, \sigma^2)$ has entropy $\tfrac12 \ln(2\pi e \sigma^2)$ and, for two Gaussians,

$$D_{\mathrm{KL}}\big(\mathcal{N}(\mu_1,\sigma_1^2)\,\|\,\mathcal{N}(\mu_2,\sigma_2^2)\big) = \ln\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac12$$

For $\mathcal{N}(0, 1)$ against $\mathcal{N}(1, 4)$ with $\sigma_2 = 2$ this is $0.4431$ nats (a Monte Carlo estimate with $10^6$ samples, seed 0, gives $0.4434$). Under a Gaussian generative model the negative log likelihood is a precision-weighted squared prediction error, $\tfrac12 \Pi\,\varepsilon^2$ plus a constant. `ContinuousActiveInference` in `code/Things/Continuous_Generic/continuous_generic.py` builds its free energy from exactly these terms: `observation_precision * prediction_error**2` for the sensory part and `belief_precisions * error**2` for each dynamics order, then descends the gradient.

## Choosing a family

| quantity | family | reason |
|----------|--------|--------|
| discrete state or outcome | categorical | finite support, normalisation is the only constraint |
| parameters of $A$, $B$, $D$ | Dirichlet | conjugate prior, pseudo-count interpretation |
| continuous states, sensory noise | Gaussian | tractable KL, Laplace approximation of the posterior |
| precision or rate parameters | Gamma | positive support, conjugate to Gaussian precision |

## Reproducing the numbers

```python
import numpy as np

rng = np.random.default_rng(0)


def entropy(p):
    p = np.asarray(p, dtype=float)
    return float(-np.sum(p[p > 0] * np.log(p[p > 0])))


print(entropy([0.5, 0.5]), entropy([0.9, 0.1]))

samples = rng.dirichlet([2.0, 3.0], size=400_000)
print(np.log(samples).mean(axis=0), (-13 / 12, -7 / 12))

mu1, s1, mu2, s2 = 0.0, 1.0, 1.0, 2.0
closed_form = np.log(s2 / s1) + (s1**2 + (mu1 - mu2) ** 2) / (2 * s2**2) - 0.5
x = rng.normal(mu1, s1, 1_000_000)
log_p = -0.5 * np.log(2 * np.pi * s1**2) - (x - mu1) ** 2 / (2 * s1**2)
log_q = -0.5 * np.log(2 * np.pi * s2**2) - (x - mu2) ** 2 / (2 * s2**2)
print(closed_form, (log_p - log_q).mean())
```

## Links

- [[probability_theory]]
- [[statistical_foundations]]
- [[exponential_family]] · [[entropy]] · [[kl_divergence]]
