---
title: Preference Learning
type: concept
status: stable
created: 2026-02-06
updated: 2026-02-07
tags:
  - preference-learning
  - C-matrix
  - prior-preferences
  - reward-learning
  - active-inference
semantic_relations:
  - type: relates
    links:
      - [[learning_mechanisms]]
      - [[decision_making]]
      - [[reinforcement_learning]]
      - [[learning_models]]
---

# Preference Learning

## Overview

Preference learning in Active Inference is the process of updating the prior preference distribution (C matrix/vector) — the agent's beliefs about which observations are desirable. Unlike reward functions in RL that are fixed, Active Inference preferences can be learned, enabling agents to discover what to value.

## Mathematical Framework

### C Vector (Prior Preferences)

```math
C_i = \ln p(o = i) \quad \text{(log-prior over preferred observations)}
```

### Preference-Driven Policy Selection

Preferences shape policy evaluation through expected free energy:

```math
G(\pi) = \sum_\tau D_{KL}[q(o_\tau|\pi) || p(o_\tau)] + \text{ambiguity}
```

where $p(o_\tau) \propto \exp(C)$ defines the preferred observation distribution.

## Preference Learning Methods

### From Reward Signals

```python
class PreferenceLearner:
    def __init__(self, n_obs, learning_rate=0.1):
        self.C = np.zeros(n_obs)
        self.lr = learning_rate
    
    def update_from_reward(self, observation, reward):
        self.C[observation] += self.lr * reward
        self.C -= self.C.mean()  # Normalize (relative preferences)
    
    def update_from_demonstration(self, demonstrated_observations):
        counts = np.bincount(demonstrated_observations, minlength=len(self.C))
        empirical = counts / counts.sum()
        self.C += self.lr * np.log(empirical + 1e-16)
```

### From Demonstrations (Inverse RL as Preference Inference)

```math
p(C | \tau_{1:N}) \propto p(C) \prod_{n=1}^N p(\tau_n | C) = p(C) \prod_n \exp(-G(\pi_n; C))
```

### Preference Evolution Dynamics

| Learning Stage | Preferences | Agent Behavior |
| --- | --- | --- |
| Tabula rasa | $C = 0$ (uniform) | Pure epistemic exploration |
| Early learning | Weak preferences | Exploration-dominant with mild goals |
| Mature | Strong preferences | Goal-directed with selective exploration |
| Expert | Refined preferences | Efficient task completion |

## Related Topics

- [[learning_mechanisms]] — General learning mechanisms
- [[decision_making]] — Decision-making
- [[reinforcement_learning]] — RL connections
- [[learning_models]] — Learning models
- [[action_selection]] — Action selection based on preferences
