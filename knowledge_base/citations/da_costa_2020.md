---
title: "Active Inference on Discrete State-Spaces: A Synthesis"
authors:
  - "Lancelot Da Costa"
  - "Thomas Parr"
  - "Noor Sajid"
  - "Sebastijan Veselic"
  - "Victorita Neacsu"
  - "Karl J. Friston"
type: citation
status: verified
created: 2025-01-01
year: 2020
journal: "Journal of Mathematical Psychology"
volume: 99
pages: 102447
doi: "10.1016/j.jmp.2020.102447"
tags:
  - active_inference
  - discrete_state_spaces
  - POMDP
  - tutorial
  - implementation
semantic_relations:
  - type: foundational_for
    links:
      - active inference discrete
      - python framework
  - type: extends
    links:
      - [[friston_2017_curiosity]]
      - [[friston_2010]]
  - type: cited_by
    links:
      - [[sajid_2021]]
      - [[smith_2022]]
      - [[parr_pezzulo_friston_2022]]
---

# Active Inference on Discrete State-Spaces: A Synthesis

## Authors
- **Lancelot Da Costa** (Imperial College London)
- **Thomas Parr** (UCL)
- **Noor Sajid** (UCL)
- **Sebastijan Veselic** (UCL)
- **Victorita Neacsu** (UCL)
- **Karl J. Friston** (UCL)

## Publication Details
- **Journal**: Journal of Mathematical Psychology
- **Year**: 2020
- **Volume**: 99
- **Pages**: 102447
- **DOI**: [10.1016/j.jmp.2020.102447](https://doi.org/10.1016/j.jmp.2020.102447)

## Abstract
This paper provides a comprehensive synthesis and tutorial for active inference on discrete state-spaces. It formulates active inference as inference on a partially observable Markov decision process (POMDP) and walks through the complete mathematical framework step by step, including state estimation, policy selection via expected free energy, parameter learning, and structure learning. It serves as the primary reference for implementing discrete active inference models.

## Key Contributions

### Complete POMDP Formulation
- **Generative Model**: Full specification with A, B, C, D, E matrices
- **State Estimation**: Variational message passing for belief updating
- **Policy Selection**: Expected free energy evaluation and softmax selection
- **Parameter Learning**: Dirichlet parameter updates for A and B matrices

### Mathematical Clarity
- **Step-by-Step Derivations**: Complete derivations of update equations
- **Clear Notation**: Consistent and accessible mathematical notation
- **Graphical Models**: Factor graph representations of the generative model
- **Algorithm Pseudocode**: Implementable algorithms for each component

### Implementation Guide
- **Matrix Formulation**: All quantities expressed as matrix operations
- **Computational Efficiency**: Practical considerations for implementation
- **Message Passing**: Neural plausible update rules
- **Software Connection**: Links to SPM implementation in MATLAB

## Core Concepts

### Generative Model (POMDP)
The generative model is specified by:
```
p(o_1:T, s_1:T, pi) = p(pi) * p(s_1) * prod_{t=1}^{T} p(o_t|s_t) * prod_{t=2}^{T} p(s_t|s_t-1, pi)
```

With matrices:
- **A**: Likelihood matrix, `p(o_t|s_t) = Cat(A)`
- **B**: Transition matrix, `p(s_t|s_t-1, u_t) = Cat(B_u)`
- **C**: Preference vector, `p(o_t) = sigma(C_t)`
- **D**: Initial state prior, `p(s_1) = Cat(D)`
- **E**: Policy prior (habits), `p(pi) = Cat(E)`

### Variational Free Energy
For state estimation:
```
F = E_q[ln q(s) - ln p(o, s|pi)]
q*(s_t) = sigma(ln A' * o_t + ln B * s_{t-1|t})  # forward message
```

### Expected Free Energy
For policy selection:
```
G(pi) = sum_{tau} [E_q[ln q(s_tau|pi) - ln p(o_tau, s_tau|pi, C)]]
       = sum_{tau} [Ambiguity + Risk - Information Gain]
```

Decomposition:
- **Pragmatic value**: `-E_q[ln p(o_tau|C)]` (goal-seeking)
- **Epistemic value**: `-E_q[H[p(o_tau|s_tau)]]` (curiosity)

### Parameter Learning
Dirichlet conjugate updates:
```
a_new = a_prior + sum_t o_t * s_t'  # update A matrix concentration parameters
b_new = b_prior + sum_t s_t * s_{t-1}'  # update B matrix concentration parameters
```

## Mathematical Formalism

### Belief Updating Algorithm
1. Initialize beliefs `s_0 = D`
2. For each observation `o_t`:
   a. Compute prediction errors: `epsilon = ln A' * o_t + ln B * s_{t-1} - ln s_t`
   b. Update beliefs: `s_t = sigma(s_t + epsilon)`
3. Evaluate policies: `G(pi) = sum_tau g(pi, tau)`
4. Select policy: `pi = sigma(-gamma * G)`
5. Execute action from selected policy

### Free Energy Gradients
State estimation via gradient descent on F:
```
ds/dt = -dF/ds = epsilon_o + epsilon_s  # prediction errors from observations and transitions
```

## Applications

### Neuroscience
- **Perceptual Inference**: State estimation in sensory processing
- **Decision Making**: Policy evaluation under uncertainty
- **Learning**: Parameter updating through experience
- **Habit Formation**: Policy prior learning

### Artificial Intelligence
- **Planning**: Multi-step lookahead with EFE
- **Exploration**: Epistemic value drives information seeking
- **Model-Based RL**: Comparison with model-based reinforcement learning
- **Robotics**: Discrete action selection for robotic agents

## Related Work

### Foundational Papers
- [[friston_2017_curiosity]] - Expected free energy introduction
- [[friston_2010]] - Free energy principle review

### Companion Tutorials
- [[smith_2022]] - Step-by-step empirical tutorial
- [[sajid_2021]] - Active inference demystified
- [[buckley_2017]] - Mathematical review (continuous case)

### Textbook
- [[parr_pezzulo_friston_2022]] - Comprehensive treatment

## Citations and Influence
This paper is the primary technical reference for discrete active inference. It has become the standard starting point for researchers implementing POMDP-based active inference models and has been widely cited in both computational neuroscience and AI literature.

## Reading Guide
1. **Generative Model**: Understand the POMDP specification
2. **State Estimation**: Variational message passing
3. **Policy Selection**: Expected free energy derivation
4. **Learning**: Parameter and structure learning
5. **Examples**: Worked simulations

---

> **Implementation Reference**: The definitive tutorial for building discrete active inference models from POMDP foundations.

---

> **Step-by-Step**: Provides complete, implementable derivations for state estimation, policy selection, and learning.

---

> **Bridge Paper**: Connects the theoretical FEP literature to practical implementation in discrete domains.
