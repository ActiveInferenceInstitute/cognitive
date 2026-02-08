---
title: "Reinforcement Learning through Active Inference"
authors:
  - "Alexander Tschantz"
  - "Beren Millidge"
  - "Anil K. Seth"
  - "Christopher L. Buckley"
type: citation
status: verified
created: 2025-01-01
year: 2020
journal: "arXiv"
doi: "10.48550/arXiv.2002.12636"
tags:
  - active_inference
  - reinforcement_learning
  - control_as_inference
  - bridging
  - AI
semantic_relations:
  - type: foundational_for
    links:
      - rl active inference
      - [[knowledge_base/cognitive/decision_making]]
  - type: extends
    links:
      - [[friston_2017_curiosity]]
      - [[sajid_2021]]
  - type: cited_by
    links:
      - [[millidge_2021]]
      - [[parr_pezzulo_friston_2022]]
---

# Reinforcement Learning through Active Inference

## Authors
- **Alexander Tschantz** (University of Sussex)
- **Beren Millidge** (University of Edinburgh)
- **Anil K. Seth** (University of Sussex)
- **Christopher L. Buckley** (University of Sussex)

## Publication Details
- **Journal**: arXiv preprint
- **Year**: 2020
- **DOI**: [10.48550/arXiv.2002.12636](https://doi.org/10.48550/arXiv.2002.12636)

## Abstract
This paper establishes formal connections between reinforcement learning (RL) and active inference, showing how standard RL algorithms can be derived within the active inference framework. The authors demonstrate that common RL objectives (expected cumulative reward) emerge as special cases of expected free energy minimization. They also show how active inference extends RL by naturally incorporating exploration through epistemic value, model uncertainty, and information gain. The paper provides both theoretical analysis and empirical demonstrations comparing the approaches.

## Key Contributions

### Formal Bridge
- **RL from Active Inference**: Derives RL objectives as special cases of EFE
- **Reward as Preference**: Reward function mapped to log prior preferences
- **Value as Free Energy**: Value function related to expected free energy
- **Policy Gradient**: Shows connections to policy gradient methods

### Extensions Beyond RL
- **Intrinsic Exploration**: Epistemic value provides principled exploration
- **Model Uncertainty**: Active inference naturally handles model uncertainty
- **Structure Learning**: Can learn model structure, not just parameters
- **Multi-Objective**: Single objective handles multiple desiderata

### Empirical Comparisons
- **Grid World Tasks**: Comparison of exploration behavior
- **Continuous Control**: Active inference in continuous action spaces
- **Sparse Reward**: Advantages in sparse reward environments
- **Stochastic Environments**: Behavior under uncertainty

## Core Concepts

### Deriving RL from Active Inference
Starting from expected free energy:
```
G(pi) = E_q[ln q(s|pi) - ln p(o, s|C)]
```

When epistemic value is zero (fully observable, no ambiguity):
```
G(pi) = -E_q[ln p(o|C)]
       = -E_q[r(o)]      (when C = exp(r))
       = -Expected Reward
```

This recovers the standard RL objective.

### Control as Inference Framework
Active inference is a form of control as inference:
```
p(pi|C) propto exp(-G(pi))         # active inference
p(a|O=1) propto exp(Q(s,a)/alpha)  # soft RL (control as inference)
```

These are formally equivalent under certain conditions.

### Where Active Inference Goes Beyond
1. **Exploration**: Epistemic value provides directed exploration
2. **Partial Observability**: Natural handling of hidden states
3. **Model Learning**: Simultaneously learn model and policy
4. **Robustness**: Prior preferences more robust than reward shaping

### Practical Advantages
- **Sparse Rewards**: Epistemic value drives exploration even without reward
- **Novel Environments**: Information-seeking in unfamiliar situations
- **Safe Exploration**: Prior preferences can encode safety constraints
- **Sample Efficiency**: Model-based nature improves sample efficiency

## Mathematical Formalism

### Expected Free Energy Decomposition
```
G(pi) = sum_tau {
  -E_q[ln p(o_tau|C)]           # Pragmatic value (reward seeking)
  +E_q[H[p(o_tau|s_tau)]]       # Ambiguity (risk aversion)
  -I_q[o_tau; s_tau|pi]          # Epistemic value (exploration)
}
```

### Connection to Soft RL
Soft actor-critic objective:
```
J(pi) = E_pi[sum_t r(s_t, a_t) + alpha * H[pi(.|s_t)]]
```

Active inference objective:
```
J_AI(pi) = -G(pi) = E_q[r(o)] + Information_Gain + ...
```

The soft RL entropy term is subsumed by the full EFE decomposition.

### Value Function Relationship
```
V_AI(s) = -min_pi G(pi|s)    # active inference "value"
V_RL(s) = max_pi E_pi[sum_t gamma^t r_t | s_0 = s]  # RL value
```

These converge when epistemic value vanishes.

## Empirical Results

### Exploration Behavior
- **RL (epsilon-greedy)**: Random exploration, slow coverage
- **Active Inference**: Directed exploration toward informative states
- **Advantage**: Faster coverage of state space, especially in sparse reward

### Sparse Reward Environments
- **RL**: Struggles without reward shaping or intrinsic motivation heuristics
- **Active Inference**: Epistemic value naturally drives exploration
- **Result**: Active inference finds rewards faster in sparse settings

## Impact and Applications

### For RL Community
- **Theoretical Foundation**: Bayesian foundation for exploration
- **Principled Exploration**: Alternative to epsilon-greedy and count-based methods
- **Unification**: Single framework for multiple RL variants

### For AI Safety
- **Constrained Behavior**: Prior preferences as behavioral constraints
- **Interpretability**: Generative models are interpretable
- **Robustness**: Less sensitive to reward misspecification

### For Robotics
- **Curiosity-Driven Exploration**: Robots that explore purposefully
- **Model-Based Control**: Sample-efficient learning for robots
- **Transfer Learning**: Generative models facilitate transfer

## Related Work

### Foundational Theory
- [[friston_2017_curiosity]] - Expected free energy
- [[sajid_2021]] - Active inference demystified and compared

### Mathematical Foundations
- [[buckley_2017]] - Mathematical review of FEP
- [[millidge_2021]] - Whence the expected free energy
- [[da_costa_2020]] - Discrete active inference

### Applications
- [[parr_pezzulo_friston_2022]] - Textbook treatment
- [[smith_2022]] - Empirical tutorial

## Citations and Influence
This paper has been influential in bridging the active inference and reinforcement learning communities. It demonstrated that active inference is not a competitor to RL but rather a generalization that subsumes standard RL as a special case while offering principled extensions for exploration and model uncertainty.

## Reading Guide
1. **Introduction**: Why bridge RL and active inference
2. **Formal Connections**: Deriving RL from EFE
3. **Extensions**: Where active inference goes beyond RL
4. **Experiments**: Empirical comparisons
5. **Discussion**: Practical implications and future work

---

> **Bridge Paper**: Establishes the formal connection between reinforcement learning and active inference.

---

> **RL as Special Case**: Shows that standard RL objectives are special cases of expected free energy minimization.

---

> **Principled Exploration**: Demonstrates how epistemic value provides directed exploration beyond epsilon-greedy heuristics.
