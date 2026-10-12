---
title: Psychophysics
type: concept
status: stable
tags:
- psychophysics
- perception
- thresholds
aliases:
  - discrimination
  - threshold_measurement
semantic_relations:
  - type: relates
    links:
      - "[[perception_processing]]"
      - "[[information_theory]]"
      - "[[active_inference]]"
---


# Psychophysics

> **Code on this page.** The Python snippets illustrate the ideas. Unless a snippet imports the
> `cognitive` package, it is not part of the package and is not run by its tests, so treat it as a
> sketch. The tested implementations are listed in [[docs/api/api_reference|the API reference]].

Psychophysics is the scientific study of the quantitative relationship between physical stimuli and perceptual experiences. It provides empirical foundations for understanding how sensory systems encode, process, and interpret environmental information. Modern psychophysics integrates signal detection theory with Bayesian inference, providing normative accounts of perception under uncertainty.

## Signal Detection Theory

### Core Concepts

#### Sensitivity (d')

The ability to discriminate signal from noise:

```math
d' = \frac{\mu_s - \mu_n}{\sigma}
```

Where:
- μ_s is the mean signal distribution
- μ_n is the mean noise distribution
- σ is the standard deviation (assumed equal)

```python
import numpy as np
from statistics import NormalDist

z = NormalDist().inv_cdf
rng = np.random.default_rng(0)
mu_n, mu_s, sigma, crit = 0.0, 1.5, 1.0, 0.75      # illustrative values
noise = rng.normal(mu_n, sigma, 20000)
signal = rng.normal(mu_s, sigma, 20000)
hit, fa = np.mean(signal > crit), np.mean(noise > crit)

print("d' from rates:", z(hit) - z(fa), " d' from definition:", (mu_s - mu_n) / sigma)
```

#### Decision Criterion (β)

Response bias in detection decisions:

```python
import numpy as np
from statistics import NormalDist

nd = NormalDist()
hit_rate, fa_rate = 0.90, 0.30                     # illustrative rates
z_hit, z_fa = nd.inv_cdf(hit_rate), nd.inv_cdf(fa_rate)

beta = nd.pdf(z_hit) / nd.pdf(z_fa)                # likelihood ratio at the criterion
beta_alt = np.exp((z_fa ** 2 - z_hit ** 2) / 2)
print(beta, beta_alt)                              # the two expressions agree
```

### Bayesian Formulation

Signal detection as Bayesian inference:

```python
import numpy as np
from statistics import NormalDist

noise, signal = NormalDist(0.0, 1.0), NormalDist(1.5, 1.0)    # illustrative equal-variance model
prior_signal = 0.3

def posterior_signal(x):
    num = prior_signal * signal.pdf(x)
    return num / (num + (1 - prior_signal) * noise.pdf(x))

# "Signal present" is reported when the posterior exceeds 1/2, i.e. when the likelihood ratio
# signal.pdf(x) / noise.pdf(x) exceeds (1 - prior_signal) / prior_signal.
for x in (0.0, 0.75, 1.5, 2.25):
    print(x, round(posterior_signal(x), 3))
```

## Psychometric Functions

### Fitting and Analysis

```python
import numpy as np

rng = np.random.default_rng(0)
logistic = lambda x, m, s: 1 / (1 + np.exp(-(x - m) / s))
levels = np.linspace(-2, 2, 9)
n_trials = 40
true_m, true_s = 0.3, 0.5                          # illustrative threshold and slope
k = rng.binomial(n_trials, logistic(levels, true_m, true_s))      # "yes" counts per level

def loglik(m, s):
    p = np.clip(logistic(levels, m, s), 1e-9, 1 - 1e-9)
    return np.sum(k * np.log(p) + (n_trials - k) * np.log(1 - p))

grid_m, grid_s = np.linspace(-1, 1, 101), np.linspace(0.1, 1.5, 71)
ll = np.array([[loglik(m, s) for s in grid_s] for m in grid_m])
i, j = np.unravel_index(ll.argmax(), ll.shape)
print("threshold, slope:", grid_m[i], grid_s[j])    # close to (0.3, 0.5) up to sampling noise
```

## Active Inference and Psychophysics

### Adaptive Psychophysics

Active inference enables adaptive experimental design:

```python
import numpy as np

rng = np.random.default_rng(0)
logistic = lambda x, m: 1 / (1 + np.exp(-(x - m) / 0.5))      # known slope, unknown threshold
thresholds = np.linspace(-2, 2, 81)
post = np.ones_like(thresholds) / len(thresholds)
candidates = np.linspace(-2, 2, 41)
H = lambda p: -np.sum(p * np.log(np.clip(p, 1e-12, 1)))
true_m = 0.6                                       # illustrative

for _ in range(30):
    def expected_entropy(x):                       # expected posterior entropy after testing stimulus x
        p1 = logistic(x, thresholds)
        py = np.sum(post * p1)
        post1 = post * p1 / py
        post0 = post * (1 - p1) / (1 - py)
        return py * H(post1) + (1 - py) * H(post0)
    x = candidates[np.argmin([expected_entropy(c) for c in candidates])]
    y = rng.random() < logistic(x, true_m)
    post = post * (logistic(x, thresholds) if y else 1 - logistic(x, thresholds))
    post /= post.sum()
print("posterior mean threshold:", np.sum(post * thresholds))
```

## Applications

### Clinical Assessment

Psychophysics in diagnosis and treatment monitoring:

- **Visual disorders**: Contrast sensitivity, visual acuity
- **Auditory disorders**: Pure tone audiometry, speech discrimination
- **Neurological assessment**: Somatosensory thresholds, temporal discrimination

### Human Factors

Psychophysics in system design:

- **Display design**: Optimal contrast ratios, font sizes
- **Warning systems**: Detectable alarm signals
- **Interface design**: Response time requirements

### Research Methodology

Advanced psychophysical techniques:

- **Magnitude estimation**: Subjective scaling of stimulus intensity
- **Cross-modal matching**: Comparing intensities across senses
- **Psychophysical reverse correlation**: Revealing perceptual templates

---

## Related Concepts

- [[perception_processing]] - General perceptual processes
- [[bayesian_inference]] - Probabilistic reasoning framework
- [[active_inference]] - Action-oriented perception
- [[signal_detection_theory]] - Decision-making under uncertainty
- [[psychometric_function]] - Mathematical models of perception


## References

- Green, D. M., & Swets, J. A. (1966). Signal Detection Theory and Psychophysics. Wiley.
- Wichmann, F. A., & Hill, N. J. (2001). The psychometric function: I. Fitting, sampling, and goodness of fit. Perception & Psychophysics, 63(8), 1293-1313.
- Knill, D. C., & Pouget, A. (2004). The Bayesian brain: The role of uncertainty in neural coding and computation. Trends in Neurosciences, 27(12), 712-719.
- Friston, K. (2010). The free-energy principle: A unified brain theory? Nature Reviews Neuroscience, 11(2), 127-138.
