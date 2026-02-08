---
title: "The Free Energy Principle for Action and Perception: A Mathematical Review"
authors:
  - "Christopher L. Buckley"
  - "Chang Sub Kim"
  - "Simon McGregor"
  - "Anil K. Seth"
type: citation
status: verified
created: 2025-01-01
year: 2017
journal: "Journal of Mathematical Psychology"
volume: 81
pages: 55-79
doi: "10.1016/j.jmp.2017.09.004"
tags:
  - free_energy
  - mathematical_review
  - perception
  - action
  - continuous
  - generalized_coordinates
semantic_relations:
  - type: foundational_for
    links:
      - [[knowledge_base/mathematics/free_energy_principle]]
      - [[knowledge_base/mathematics/generalized_coordinates]]
  - type: extends
    links:
      - [[friston_2006]]
      - [[friston_2010]]
  - type: cited_by
    links:
      - [[millidge_2021]]
      - [[da_costa_2021_bayesian]]
      - [[parr_pezzulo_friston_2022]]
---

# The Free Energy Principle for Action and Perception: A Mathematical Review

## Authors
- **Christopher L. Buckley** (University of Sussex)
- **Chang Sub Kim** (Chungbuk National University)
- **Simon McGregor** (University of Sussex)
- **Anil K. Seth** (University of Sussex)

## Publication Details
- **Journal**: Journal of Mathematical Psychology
- **Year**: 2017
- **Volume**: 81
- **Pages**: 55-79
- **DOI**: [10.1016/j.jmp.2017.09.004](https://doi.org/10.1016/j.jmp.2017.09.004)

## Abstract
This paper provides a detailed mathematical review of the free energy principle for action and perception in continuous time. It carefully derives the key equations from first principles, including the Laplace approximation to variational free energy, generalized coordinates of motion, and the complete perception-action loop. The review aims to make the mathematical foundations of the FEP accessible to researchers by providing step-by-step derivations that are often omitted or compressed in Friston's original papers.

## Key Contributions

### Mathematical Accessibility
- **Step-by-Step Derivations**: Complete derivations often missing from original papers
- **Clear Notation**: Consistent and explicit mathematical notation
- **Assumptions Made Explicit**: All simplifying assumptions clearly stated
- **Self-Contained**: Requires only basic probability theory and calculus

### Continuous Formulation
- **Laplace Approximation**: Free energy under Gaussian assumptions
- **Generalized Coordinates**: Extended states including velocities, accelerations, etc.
- **Generalized Filtering**: State estimation in generalized coordinates
- **Active Inference**: Action as minimizing free energy in continuous time

### Critical Assessment
- **Assumptions Examined**: Which approximations are necessary vs convenient
- **Limitations Identified**: Where the framework faces challenges
- **Open Questions**: Unresolved mathematical issues highlighted
- **Connection to ML**: Links to variational inference in machine learning

## Core Concepts

### Laplace-Encoded Free Energy
Under the Laplace (Gaussian) approximation:
```
F = -ln p(y_tilde|mu_tilde, theta) + 1/2 * ln |Pi_mu| + const
```

Where:
- `y_tilde`: Observations in generalized coordinates
- `mu_tilde`: Expected hidden states in generalized coordinates
- `theta`: Model parameters
- `Pi_mu`: Precision of the posterior

### Generalized Coordinates of Motion
States are extended to include temporal derivatives:
```
x_tilde = (x, x', x'', x''', ...)
```

Where:
- `x`: Position (zeroth order)
- `x'`: Velocity (first order)
- `x''`: Acceleration (second order)
- Higher orders encode increasingly distal temporal structure

### Perception-Action Loop
The complete perception-action cycle:
```
Perception: dmu/dt = D*mu - dF/dmu  # state estimation
Action:     da/dt = -dF/da            # active inference
Learning:   dtheta/dt = -dF/dtheta    # parameter learning
```

Where D is the differential operator in generalized coordinates.

### Prediction Error Formulation
Free energy gradients reduce to prediction errors:
```
dF/dmu = -epsilon_y * Pi_y * dg/dmu - epsilon_mu * Pi_mu
```

Where:
- `epsilon_y = y - g(mu)`: Sensory prediction error
- `epsilon_mu = D*mu - f(mu)`: Dynamic prediction error
- `Pi_y, Pi_mu`: Precision matrices

## Mathematical Formalism

### Generative Model
The generative model in continuous time:
```
dy/dt = g(x, v) + z_y    # observation function + noise
dx/dt = f(x, v) + z_x    # state dynamics + noise
```

Where:
- `g`: Observation function mapping hidden states to sensory data
- `f`: State transition function
- `v`: Causes from higher hierarchical levels
- `z`: Random fluctuations

### Hierarchical Extension
At each level `i` of the hierarchy:
```
y_i = g_i(x_i, v_i) + z_i
v_i = x_{i+1}
```

Prediction errors propagate bottom-up; predictions flow top-down.

### Precision Dynamics
Precision parameters evolve to minimize free energy:
```
dgamma/dt = -dF/dgamma
```

This implements attention and uncertainty estimation.

## Impact and Applications

### For Researchers
- **Entry Point**: Best mathematical introduction to continuous FEP
- **Reference**: Standard reference for equations and derivations
- **Implementation**: Sufficient detail for numerical implementation

### For Machine Learning
- **Variational Methods**: Connections to variational autoencoders
- **Sequential Inference**: Generalized filtering as online inference
- **Control**: Active inference as a control algorithm

### For Neuroscience
- **Neural Implementation**: How equations map to neural circuits
- **Predictive Coding**: Complete derivation of predictive coding scheme
- **Motor Control**: Active inference for continuous motor behavior

## Related Work

### Foundational Papers
- [[friston_2006]] - Original FEP formulation
- [[friston_2010]] - Unified brain theory review
- [[friston_2009]] - Predictive coding under FEP

### Companion Reviews
- [[da_costa_2020]] - Discrete state-space synthesis
- [[sajid_2021]] - Active inference compared

### Extensions
- [[millidge_2021]] - Expected free energy derivation
- [[da_costa_2021_bayesian]] - Bayesian mechanics

## Citations and Influence
This review is widely regarded as the most accessible mathematical treatment of the continuous free energy principle. It has been cited extensively by researchers who found Friston's original derivations difficult to follow, and it serves as the standard mathematical reference for the continuous formulation of the FEP.

## Reading Guide
1. **Introduction**: Motivation and scope
2. **Variational Free Energy**: Core derivation
3. **Generalized Coordinates**: Extended state formulation
4. **Perception**: State estimation equations
5. **Action**: Active inference derivation
6. **Hierarchical Models**: Multi-level extension
7. **Discussion**: Assumptions, limitations, open questions

---

> **Mathematical Clarity**: The most accessible and complete mathematical treatment of the continuous free energy principle.

---

> **Step-by-Step**: Provides derivations that are often compressed or omitted in original papers.

---

> **Critical Assessment**: Honestly examines assumptions, limitations, and open questions in the FEP formalism.
