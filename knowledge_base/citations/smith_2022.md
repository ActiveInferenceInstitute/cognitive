---
title: "A Step-by-Step Tutorial on Active Inference and Its Application to Empirical Data"
authors:
  - "Ryan Smith"
  - "Karl J. Friston"
  - "Christopher J. Whyte"
type: citation
status: verified
created: 2025-01-01
year: 2022
journal: "Journal of Mathematical Psychology"
volume: 107
pages: 102632
doi: "10.1016/j.jmp.2021.102632"
tags:
  - active_inference
  - tutorial
  - empirical
  - model_fitting
  - computational_modeling
semantic_relations:
  - type: foundational_for
    links:
      - python framework
      - active inference discrete
  - type: extends
    links:
      - [[da_costa_2020]]
      - [[friston_2017_curiosity]]
  - type: cited_by
    links:
      - [[parr_pezzulo_friston_2022]]
---

# A Step-by-Step Tutorial on Active Inference and Its Application to Empirical Data

## Authors
- **Ryan Smith** (Laureate Institute for Brain Research)
- **Karl J. Friston** (UCL)
- **Christopher J. Whyte** (Macquarie University)

## Publication Details
- **Journal**: Journal of Mathematical Psychology
- **Year**: 2022
- **Volume**: 107
- **Pages**: 102632
- **DOI**: [10.1016/j.jmp.2021.102632](https://doi.org/10.1016/j.jmp.2021.102632)

## Abstract
This paper provides a practical, step-by-step tutorial on how to build active inference models and fit them to empirical behavioral data. It covers the complete pipeline from specifying a generative model, through simulating agent behavior, to fitting the model to experimental data using variational Bayes and comparing models using Bayesian model comparison. The tutorial uses a concrete experimental paradigm (a two-armed bandit task) to demonstrate each step, making the framework accessible to experimentalists.

## Key Contributions

### Practical Implementation Guide
- **Model Specification**: How to define generative models for experiments
- **Simulation**: Running active inference simulations to generate behavior
- **Parameter Estimation**: Fitting models to behavioral data
- **Model Comparison**: Comparing alternative models using free energy

### Empirical Application
- **Two-Armed Bandit**: Complete worked example with real paradigm
- **Behavioral Data**: Fitting to choice and reaction time data
- **Individual Differences**: Estimating subject-specific parameters
- **Group Comparisons**: Comparing populations using model parameters

### Complete Pipeline
- **Step 1**: Define the generative model (A, B, C, D matrices)
- **Step 2**: Simulate behavior under the model
- **Step 3**: Fit the model to observed data
- **Step 4**: Perform model comparison
- **Step 5**: Interpret parameter estimates

## Core Concepts

### Building a Generative Model
For a two-armed bandit task:
```
A: Observation likelihood
   - Maps hidden states (reward locations) to observations (reward/no reward)

B: Transition matrix
   - Models state transitions under actions (choose left/right)

C: Preferences
   - Prior preference for reward outcomes

D: Initial state prior
   - Prior belief about which arm is rewarding
```

### Simulation
Given the generative model, simulate an active inference agent:
1. Initialize beliefs using D
2. For each trial:
   a. Evaluate expected free energy for each policy
   b. Select policy via softmax
   c. Execute action, receive observation
   d. Update beliefs using variational message passing
   e. Update parameters using Dirichlet learning

### Model Fitting
Fit models to observed behavior using variational Bayes:
```
F = E_q[ln p(y, theta|m)] - E_q[ln q(theta)]
theta_MAP = argmax_theta p(theta|y, m)
```

Where:
- `y`: Observed behavioral data (choices, reaction times)
- `theta`: Model parameters to estimate
- `m`: Model identity

### Model Comparison
Compare models using variational free energy as model evidence:
```
ln p(y|m) approx -F(m)
BMS: Compare F across models to determine best model
```

## Mathematical Formalism

### Parameter Estimation
For each participant, estimate parameters by minimizing:
```
F(theta) = -E_q[ln p(choices|theta, m)] - E_q[ln p(theta)]
```

Key parameters typically estimated:
- **alpha**: Action precision (inverse temperature)
- **eta**: Learning rate for parameter updates
- **beta**: Prior precision over policies

### Posterior Predictive Checks
Validate model fit by:
1. Simulate behavior using estimated parameters
2. Compare simulated and actual choice patterns
3. Assess recovery of known parameters in simulations

## Practical Considerations

### Common Pitfalls
- **Identifiability**: Ensure parameters are estimable from data
- **Prior Sensitivity**: Check robustness to prior specifications
- **Model Recovery**: Verify parameters can be recovered from simulated data
- **Overfitting**: Use model comparison to avoid overly complex models

### Software and Code
- **MATLAB/SPM**: Original implementation
- **Python (pymdp)**: Open-source Python implementation
- **Code Availability**: Tutorial code provided with the paper

## Impact and Applications

### For Experimentalists
- **Accessible**: Written for researchers without deep theoretical background
- **Practical**: Directly applicable to experimental paradigms
- **Reproducible**: Complete code and data provided

### Computational Psychiatry
- **Clinical Parameters**: Map model parameters to clinical symptoms
- **Biomarkers**: Use computational parameters as disease markers
- **Treatment Prediction**: Predict treatment response from model parameters

### Cognitive Psychology
- **Decision Making**: Model choice behavior under uncertainty
- **Learning**: Quantify learning rates and strategies
- **Individual Differences**: Characterize cognitive phenotypes

## Related Work

### Theoretical Foundations
- [[da_costa_2020]] - Mathematical synthesis of discrete active inference
- [[friston_2017_curiosity]] - Expected free energy theory
- [[parr_pezzulo_friston_2022]] - Comprehensive textbook

### Related Tutorials
- [[sajid_2021]] - Active inference demystified
- [[buckley_2017]] - Mathematical review (continuous)

### Applications
- [[hesp_2021]] - Deep active inference and affect

## Citations and Influence
This tutorial has become a key practical resource for researchers wanting to apply active inference to empirical data. It bridges the gap between theoretical papers and hands-on implementation, making the framework accessible to experimentalists in psychology, psychiatry, and cognitive neuroscience.

## Reading Guide
1. **Introduction**: Why use active inference for empirical modeling
2. **Model Specification**: Defining the generative model
3. **Simulation**: Generating synthetic behavior
4. **Model Fitting**: Estimating parameters from data
5. **Model Comparison**: Selecting the best model
6. **Discussion**: Practical considerations and extensions

---

> **Practical Tutorial**: The most accessible hands-on guide for fitting active inference models to empirical data.

---

> **Complete Pipeline**: Covers the full workflow from model specification through simulation to parameter estimation and model comparison.

---

> **For Experimentalists**: Written specifically for researchers who want to apply active inference to their experiments.
