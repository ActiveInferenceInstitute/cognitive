---
title: "Active Inference, Curiosity and Insight"
authors:
  - "Karl J. Friston"
  - "Marco Lin"
  - "Christopher D. Frith"
  - "Giovanni Pezzulo"
  - "J. Allan Hobson"
  - "Sasha Ondobaka"
type: citation
status: verified
created: 2025-01-01
year: 2017
journal: "Neural Computation"
volume: 29
issue: 10
pages: 2633-2683
doi: "10.1162/neco_a_00999"
tags:
  - active_inference
  - curiosity
  - expected_free_energy
  - epistemic_value
  - exploration
semantic_relations:
  - type: foundational_for
    links:
      - [[knowledge_base/mathematics/expected_free_energy]]
      - curiosity
      - exploration
  - type: extends
    links:
      - [[friston_2010]]
      - [[friston_2013]]
  - type: cited_by
    links:
      - [[da_costa_2020]]
      - [[sajid_2021]]
      - [[millidge_2021]]
---

# Active Inference, Curiosity and Insight

## Authors
- **Karl J. Friston** (Wellcome Trust Centre for Neuroimaging, UCL)
- **Marco Lin**
- **Christopher D. Frith**
- **Giovanni Pezzulo**
- **J. Allan Hobson**
- **Sasha Ondobaka**

## Publication Details
- **Journal**: Neural Computation
- **Year**: 2017
- **Volume**: 29
- **Issue**: 10
- **Pages**: 2633-2683
- **DOI**: [10.1162/neco_a_00999](https://doi.org/10.1162/neco_a_00999)

## Abstract
This paper formalizes curiosity and insight within the active inference framework by introducing the expected free energy (EFE) as the objective function for policy selection. The EFE naturally decomposes into pragmatic (goal-directed) and epistemic (information-seeking) components, providing a principled account of how agents balance exploitation and exploration. The paper demonstrates how curiosity emerges from epistemic value and how insight corresponds to large reductions in free energy during model updating.

## Key Contributions

### Expected Free Energy (EFE)
- **Formal Definition**: The free energy expected under predicted outcomes for a given policy
- **Pragmatic Value**: Preferences over outcomes (goal-seeking, exploitation)
- **Epistemic Value**: Information gain about hidden states (curiosity, exploration)
- **Natural Exploration**: Exploration emerges from information-seeking, not random noise

### Curiosity as Epistemic Value
- **Intrinsic Motivation**: Agents seek information-rich observations
- **Uncertainty Resolution**: Curiosity drives reduction of uncertainty about hidden states
- **Salience**: Epistemic value explains attentional salience
- **Novelty Seeking**: New observations are sought when they resolve ambiguity

### Insight as Free Energy Reduction
- **Aha Moments**: Sudden model updates that dramatically reduce free energy
- **Belief Revision**: Large shifts in posterior beliefs after key observations
- **Learning Transitions**: Phase transitions in understanding
- **Gestalt Shifts**: Perceptual reorganization through model updating

## Core Concepts

### Expected Free Energy
The expected free energy for a policy pi is:

```
G(pi) = E_q[ln q(s_tau|pi) - ln p(o_tau, s_tau|pi)]
```

This decomposes into:

```
G(pi) = -E_q[ln p(o_tau|C)]  +  E_q[H[p(o_tau|s_tau)]]  -  E_q[KL[q(s_tau|o_tau,pi) || q(s_tau|pi)]]
```

Where:
- First term: **Pragmatic value** (negative, preferences over outcomes)
- Second term: **Ambiguity** (expected uncertainty about observations given states)
- Third term: **Epistemic value** (negative, information gain about states)

### Policy Selection
Policies are selected by their expected free energy:
```
pi* = argmin_pi G(pi)
P(pi) = sigma(-gamma * G(pi))  # softmax policy selection
```

Where gamma is a precision parameter (inverse temperature) governing exploration-exploitation.

### Exploration-Exploitation Balance
- **High epistemic value**: Agent explores to gain information
- **High pragmatic value**: Agent exploits to achieve preferences
- **Natural balance**: Both terms compete under the same objective
- **No epsilon-greedy**: Exploration is principled, not random

## Mathematical Formalism

### Decomposition of EFE
The EFE can also be written as:
```
G(pi) = E_q[-ln p(o_tau|C)] - E_q[I(o_tau; s_tau|pi)]
```

Where:
- `p(o_tau|C)`: Prior preferences over observations (from matrix C)
- `I(o_tau; s_tau|pi)`: Mutual information between observations and states

### Information Gain
The epistemic value is the expected information gain:
```
Epistemic Value = E_q[KL[q(s_tau|o_tau, pi) || q(s_tau|pi)]]
                = E_q[D_KL(posterior || prior)]
```

This drives the agent to seek observations that maximally update its beliefs.

### Relationship to Other Objectives
- **KL Control**: EFE subsumes KL control as a special case
- **Risk-Sensitive Control**: Pragmatic value relates to risk
- **Bayesian Optimal Experiment Design**: Epistemic value as optimal design
- **Intrinsic Motivation**: Connections to empowerment and surprise

## Applications

### Cognitive Science
- **Visual Search**: Saccadic eye movements as epistemic foraging
- **Scientific Inquiry**: Hypothesis testing as active inference
- **Problem Solving**: Insight through belief revision
- **Creativity**: Novel solutions through exploratory inference

### Neuroscience
- **Dopamine**: Encodes precision of policy selection
- **Anterior Cingulate**: Evaluates epistemic vs pragmatic value
- **Hippocampus**: Novelty detection as epistemic salience
- **Prefrontal Cortex**: Policy evaluation and selection

### Artificial Intelligence
- **Active Learning**: Principled sample selection
- **Reinforcement Learning**: Intrinsic reward from information gain
- **Robotics**: Curiosity-driven exploration in novel environments
- **Scientific Discovery**: Automated experiment design

## Related Work

### Foundational Papers
- [[friston_2010]] - Free energy principle review
- [[friston_2013]] - Life as we know it

### Extensions
- [[da_costa_2020]] - Active inference on discrete state-spaces
- [[sajid_2021]] - Active inference demystified
- [[millidge_2021]] - Whence the expected free energy

### Related Concepts
- [[parr_friston_2017]] - Working memory and salience
- [[smith_2022]] - Step-by-step tutorial

## Citations and Influence
This paper is foundational for understanding how active inference handles exploration and exploitation. The expected free energy has become the central objective function in discrete active inference models and provides the formal basis for curiosity, salience, and insight in the FEP literature. It has been cited extensively in both neuroscience and AI research.

## Reading Guide
1. **Introduction**: Motivation for formalizing curiosity
2. **Expected Free Energy**: Definition and decomposition
3. **Curiosity**: Epistemic value as intrinsic motivation
4. **Insight**: Sudden free energy reduction
5. **Simulations**: Demonstrations of exploratory behavior

---

> **Expected Free Energy**: Introduces the EFE as the central objective for policy selection in active inference.

---

> **Curiosity Formalized**: Provides a principled account of exploration through epistemic value rather than random noise.

---

> **Exploration-Exploitation**: Resolves the exploration-exploitation dilemma through a single objective function.
