---
title: Information Theory in Control
type: concept
status: stable
created: 2026-02-06
updated: 2026-02-07
tags:
  - information-theory
  - control-theory
  - rate-distortion
  - channel-capacity
  - active-inference
semantic_relations:
  - type: relates
    links:
      - [[active_inference_for_control]]
      - [[free_energy_principle]]
      - [[knowledge_base/mathematics/information_theory]]
      - [[advanced_control]]
---

# Information Theory in Control

## Overview

Information theory provides the mathematical language connecting control and inference. Active Inference naturally incorporates information-theoretic quantities: the free energy is an information-theoretic bound, epistemic value is mutual information, and precision is an inverse entropy. This page covers key information-theoretic concepts in control.

## Key Quantities

### Mutual Information and Control

```math
I(S; O) = H(O) - H(O|S) = \sum_{s,o} p(s,o) \ln \frac{p(s,o)}{p(s)p(o)}
```

High $I(S;O)$ means observations are informative about states — a prerequisite for effective control.

### Rate-Distortion Theory

The minimum information rate needed for control to within distortion $D$:

```math
R(D) = \min_{p(\hat{s}|s): \mathbb{E}[d(s,\hat{s})] \leq D} I(S; \hat{S})
```

This sets a fundamental limit: controllers cannot outperform the information available through their sensory channel.

### Information Cost of Control

```math
C_{\text{info}} = \sum_t I(U_t; S_{0:t}, O_{0:t}) \quad \text{(bits of information used by policy)}
```

Information-theoretic bounded rationality constrains agents to use limited information for control.

## Connections to Active Inference

| Information Concept | Active Inference Component | Role |
| --- | --- | --- |
| Mutual information $I(s;o)$ | Epistemic value | Drives exploration |
| Entropy $H(q(s))$ | Beliefs uncertainty | Motivates info-gathering |
| KL divergence $D_{KL}$ | Free energy bound | Forces beliefs toward posterior |
| Channel capacity | Sensory bandwidth | Limits observation model |
| Rate-distortion | Compression | Generative model complexity |
| Transfer entropy | Causal influence | Effective connectivity |

## Implementation

```python
def compute_info_metrics(A, beliefs, observations):
    # Mutual information between states and observations
    joint = A * beliefs[:, np.newaxis]
    marginal_o = joint.sum(axis=0)
    MI = np.sum(joint * np.log(joint / (beliefs[:, np.newaxis] * marginal_o[np.newaxis, :] + 1e-16) + 1e-16))
    
    # State entropy
    H_s = -np.sum(beliefs * np.log(beliefs + 1e-16))
    
    # Observation entropy 
    H_o = -np.sum(marginal_o * np.log(marginal_o + 1e-16))
    
    return {'MI': MI, 'H_states': H_s, 'H_obs': H_o}
```

## Related Topics

- [[active_inference_for_control]] — Active Inference in control systems
- [[free_energy_principle]] — Free energy principle
- [[advanced_control]] — Advanced control methods
- [[knowledge_base/mathematics/information_theory]] — Information theory foundations
