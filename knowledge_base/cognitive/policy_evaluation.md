---
title: Policy Evaluation
type: concept
status: stable
created: 2026-02-06
updated: 2026-02-07
tags:
  - policy-evaluation
  - expected-free-energy
  - planning
  - decision-making
  - active-inference
semantic_relations:
  - type: relates
    links:
      - [[planning_as_inference]]
      - [[decision_making]]
      - [[policy_selection]]
      - [[action_selection]]
      - [[knowledge_base/mathematics/expected_free_energy]]
---

# Policy Evaluation

## Overview

Policy evaluation in Active Inference computes the expected free energy $G(\pi)$ for each candidate policy, providing a principled score that balances pragmatic value (achieving goals) with epistemic value (resolving uncertainty). This evaluation determines the probability of selecting each policy.

## Expected Free Energy

### Definition

```math
G(\pi) = \sum_{\tau=t+1}^{T} G(\pi, \tau)
```

### Per-Timestep Decomposition

```math
G(\pi, \tau) = \underbrace{\mathbb{E}_{q(s_\tau|\pi)}[H[p(o_\tau|s_\tau)]]}_{\text{ambiguity (sensory uncertainty)}} + \underbrace{D_{KL}[q(o_\tau|\pi) || p(o_\tau)]}_{\text{risk (preference divergence)}}
```

Equivalently:

```math
G(\pi, \tau) = \underbrace{-I[o_\tau; s_\tau | \pi]}_{\text{negative info gain}} + \underbrace{\mathbb{E}_{q(o_\tau|\pi)}[-\ln p(o_\tau)]}_{\text{expected surprise}}
```

## Implementation

```python
def evaluate_policy(A, B, C, beliefs, policy, T_horizon):
    G = 0.0
    predicted_states = beliefs.copy()
    
    for t in range(T_horizon):
        action = policy[t] if t < len(policy) else policy[-1]
        
        # Predict future states
        predicted_states = B[action] @ predicted_states
        
        # Predict future observations
        predicted_obs = A @ predicted_states
        
        # Ambiguity: expected entropy of observations given states
        H_obs_given_s = -np.sum(predicted_states * 
            np.sum(A * np.log(A + 1e-16), axis=0))
        
        # Risk: KL divergence from preferred observations
        preferred = softmax(C)
        risk = np.sum(predicted_obs * np.log(
            (predicted_obs + 1e-16) / (preferred + 1e-16)))
        
        G += H_obs_given_s + risk
    
    return G
```

### Policy Posterior

```math
q(\pi) = \sigma(-\gamma \cdot G(\pi)) = \frac{\exp(-\gamma G(\pi))}{\sum_{\pi'} \exp(-\gamma G(\pi'))}
```

### $\gamma$ Effect on Behavior

| $\gamma$ (precision) | Behavior | Character |
| --- | --- | --- |
| $\gamma \to 0$ | Random | Maximum exploration |
| $\gamma \approx 1$ | Softmax | Balanced |
| $\gamma \to \infty$ | Argmax | Maximum exploitation |

## Related Topics

- [[planning_as_inference]] — Planning as inference
- [[policy_selection]] — Policy selection mechanisms
- [[decision_making]] — Decision-making processes
- [[action_selection]] — Action selection
- [[knowledge_base/mathematics/expected_free_energy]] — EFE derivation
