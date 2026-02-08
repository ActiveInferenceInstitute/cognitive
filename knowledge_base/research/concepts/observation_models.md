---
title: Observation Models
type: concept
status: stable
created: 2024-02-12
updated: 2026-02-07
tags: [observation, likelihood, sensory, research, generative_model]
semantic_relations:
  - type: relates
    links: [[knowledge_base/cognitive/observation_model]], [[hidden_states]], [[knowledge_base/mathematics/active_inference_pomdp]], [[information_metrics]]]
---

# Observation Models

Research perspectives on observation models (likelihood functions) in Active Inference, covering design considerations, learning dynamics, multi-modal fusion, and empirical evaluation.

## Design Considerations

### Observation Model Properties

| Property | Definition | Impact on Inference |
| --- | --- | --- |
| Informativeness | $I(O; S) = H[O] - H[O|S]$ | State discriminability |
| Ambiguity | $H[O|S] = -\sum_j p(s_j)\sum_i A_{ij}\ln A_{ij}$ | Noise level |
| Sparsity | Fraction of near-zero entries | Computational efficiency |
| Modularity | Block-diagonal structure | Factored inference |
| Symmetry | $A_{ij} = A_{ji}$ | Indistinguishable states |

### Observation Model Taxonomy

```mermaid
graph TD
    subgraph "Observation Model Types"
        D[Deterministic: A is permutation] --> I[Informative: I(O;S) is high]
        N[Noisy: H(O|S) > 0] --> P[Partial: some states unobservable]
        MM[Multi-Modal: A factored across modalities]
        F[Factored: A block-diagonal per state factor]
    end
    style D fill:#bfb,stroke:#333
    style N fill:#fbb,stroke:#333
```

### Multi-Modal Observations

Models with multiple sensory modalities use factored observation likelihoods:

```math
p(o_t | s_t) = \prod_{m=1}^{M} p(o_t^{(m)} | s_t) = \prod_{m=1}^{M} \text{Cat}(o_t^{(m)} | A^{(m)} s_t)
```

This enables sensory fusion where information from one modality compensates for ambiguity in another.

## Learning Dynamics

### Dirichlet Learning

```python
def learn_observation_model(a_prior, observations, states, learning_rate=1.0):
    """Learn A matrix via Dirichlet concentration parameter updates."""
    a = a_prior.copy()
    for o, s in zip(observations, states):
        a[o, s] += learning_rate
    A = a / a.sum(axis=0, keepdims=True)
    return A, a

def compute_epistemic_value(A, beliefs):
    """Compute expected information gain from observations."""
    predicted_obs = A @ beliefs
    H_obs = -np.sum(predicted_obs * np.log(predicted_obs + 1e-16))
    H_cond = -np.sum(beliefs * np.sum(A * np.log(A + 1e-16), axis=0))
    return H_obs - H_cond  # Mutual information I(O;S)
```

### Learning Curve Properties

| Phase | Trials | $a_{ij}$ Magnitude | Model Quality |
| --- | --- | --- | --- |
| Initial | 0-10 | Prior-dominated | Low accuracy |
| Learning | 10-100 | Data-influenced | Improving |
| Asymptotic | 100+ | Data-dominated | Near-optimal |

## Evaluation Criteria

1. **Prediction accuracy**: $\frac{1}{T}\sum_t \mathbb{1}[\hat{o}_t = o_t]$
2. **Log-likelihood**: $\frac{1}{T}\sum_t \ln p(o_t | \hat{s}_t)$
3. **Information transfer**: $I(O; S)$ under the learned model
4. **Calibration**: Agreement between predicted and empirical observation frequencies

## Related Topics

- [[knowledge_base/cognitive/observation_model]] — A matrix specification
- [[hidden_states]] — Hidden state inference
- [[information_metrics]] — Information-theoretic metrics
- [[knowledge_base/mathematics/active_inference_pomdp]] — POMDP formulation
