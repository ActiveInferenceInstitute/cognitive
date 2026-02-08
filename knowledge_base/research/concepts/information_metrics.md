---
title: Information Metrics
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [information_theory, metrics, entropy, mutual_information]
semantic_relations:
  - type: relates
    links: [[knowledge_base/mathematics/information_theory]], [[knowledge_base/mathematics/entropy]], [[knowledge_base/mathematics/mutual_information]], [[knowledge_base/mathematics/kl_divergence]]]
---

# Information Metrics

Quantitative measures from information theory used to evaluate Active Inference agents and generative models, including entropy, mutual information, KL divergence, and transfer entropy.

## Core Metrics

### Entropy

```math
H[X] = -\sum_x p(x) \ln p(x)
```

Measures the average surprise or uncertainty in a random variable.

### Mutual Information

```math
I(X; Y) = H[X] - H[X|Y] = \sum_{x,y} p(x,y) \ln \frac{p(x,y)}{p(x)p(y)}
```

### KL Divergence

```math
D_{KL}[q||p] = \sum_x q(x) \ln \frac{q(x)}{p(x)}
```

### Transfer Entropy

```math
T_{X \to Y} = \sum p(y_{t+1}, y_t^{(k)}, x_t^{(l)}) \ln \frac{p(y_{t+1}|y_t^{(k)}, x_t^{(l)})}{p(y_{t+1}|y_t^{(k)})}
```

## Application to Active Inference

| Metric | Application | Interpretation |
| --- | --- | --- |
| $H[q(s)]$ | Belief entropy | Agent's uncertainty |
| $I(O; S)$ | Observation informativeness | Sensory channel quality |
| $D_{KL}[q||p]$ | Model fit | Free energy complexity |
| $G_{epistemic}$ | Expected info gain | Exploration drive |

## Implementation

```python
def compute_information_metrics(beliefs, observations, model):
    H_beliefs = -np.sum(beliefs * np.log(beliefs + 1e-16))
    MI = mutual_information(observations, beliefs, model.A)
    KL = np.sum(beliefs * np.log((beliefs + 1e-16) / (model.D + 1e-16)))
    return {'entropy': H_beliefs, 'mutual_info': MI, 'kl_divergence': KL}
```

## Related Topics

- [[knowledge_base/mathematics/information_theory]] — Information theory basics
- [[knowledge_base/mathematics/entropy]] — Entropy measures
- [[knowledge_base/mathematics/mutual_information]] — Mutual information
- [[knowledge_base/mathematics/kl_divergence]] — KL divergence
