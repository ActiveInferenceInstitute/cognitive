---
title: "Active Inference: The Free Energy Principle in Mind, Brain, and Behavior"
authors:
  - "Thomas Parr"
  - "Giovanni Pezzulo"
  - "Karl J. Friston"
type: citation
status: verified
created: 2025-01-01
year: 2022
journal: "MIT Press"
doi: "10.7551/mitpress/12441.001.0001"
tags:
  - active_inference
  - textbook
  - free_energy
  - generative_models
  - bayesian_inference
  - comprehensive
semantic_relations:
  - type: foundational_for
    links:
      - [[knowledge_base/mathematics/free_energy_principle]]
      - [[knowledge_base/cognitive/active_inference]]
      - [[knowledge_base/mathematics/expected_free_energy]]
  - type: extends
    links:
      - [[friston_2010]]
      - [[friston_2017_curiosity]]
      - [[da_costa_2020]]
  - type: cited_by
    links: []
---

# Active Inference: The Free Energy Principle in Mind, Brain, and Behavior

## Authors
- **Thomas Parr** (University College London)
- **Giovanni Pezzulo** (Institute of Cognitive Sciences and Technologies, CNR)
- **Karl J. Friston** (Wellcome Trust Centre for Neuroimaging, UCL)

## Publication Details
- **Publisher**: MIT Press
- **Year**: 2022
- **DOI**: [10.7551/mitpress/12441.001.0001](https://doi.org/10.7551/mitpress/12441.001.0001)

## Abstract
This is the definitive textbook on active inference and the free energy principle. It provides a comprehensive, pedagogically organized treatment of the theory from first principles through to advanced applications. The book covers the mathematical foundations, neural process theories, discrete and continuous formulations, and applications across neuroscience, psychology, and artificial intelligence. It serves as both an introduction for newcomers and a reference for researchers.

## Key Contributions

### Comprehensive Pedagogy
- **First Principles**: Builds the theory from basic probability theory
- **Progressive Complexity**: Moves from simple to complex formulations
- **Worked Examples**: Detailed simulations and demonstrations
- **Unified Notation**: Consistent mathematical notation throughout

### Discrete and Continuous Formulations
- **Discrete State-Spaces**: Partially observable Markov decision processes (POMDPs)
- **Continuous State-Spaces**: Stochastic differential equations and generalized coordinates
- **Mixed Models**: Combining discrete and continuous variables
- **Deep Temporal Models**: Hierarchical policies over multiple time scales

### Neural Process Theory
- **Predictive Coding**: Continuous state inference in cortical hierarchies
- **Belief Propagation**: Discrete state inference in neural circuits
- **Precision Weighting**: Neuromodulation as gain control
- **Synaptic Plasticity**: Learning as parameter optimization

### Applications
- **Perception**: Visual, auditory, and somatosensory inference
- **Motor Control**: Action as active inference
- **Decision Making**: Policy selection under uncertainty
- **Social Cognition**: Inference about other agents
- **Psychopathology**: Computational psychiatry applications

## Core Concepts

### Generative Models
The book organizes active inference around generative models:
```
p(o, s, pi, theta) = p(o|s) * p(s|s', pi) * p(pi) * p(theta)
```

Where:
- `o`: Observations (outcomes)
- `s`: Hidden states
- `pi`: Policies (sequences of actions)
- `theta`: Parameters (learned quantities)

### Free Energy Functionals
Three key free energy quantities:
1. **Variational Free Energy (F)**: Bound on surprise for current observations
2. **Expected Free Energy (G)**: Anticipated free energy for future policies
3. **Bayesian Model Evidence**: Evidence for the generative model itself

### The POMDP Framework
Discrete active inference uses POMDPs with:
- **A matrix**: Likelihood mapping states to observations
- **B matrix**: Transition probabilities under actions
- **C vector**: Prior preferences over observations
- **D vector**: Prior beliefs about initial states
- **E vector**: Prior beliefs about policies (habits)

### Message Passing
Inference proceeds through message passing:
```
s_posterior = sigma(ln A' * o + ln B * s_prior)  # state estimation
G(pi) = sum_tau [o_bar' * (ln o_bar - ln C) + ...]  # policy evaluation
```

## Chapter Summary

### Part I: Foundations
- **Chapter 1**: The free energy principle and its motivation
- **Chapter 2**: Generative models and variational inference
- **Chapter 3**: Inference, learning, and model selection

### Part II: Discrete Models
- **Chapter 4**: Active inference in discrete time
- **Chapter 5**: Expected free energy and policy selection
- **Chapter 6**: Structure learning and deep temporal models

### Part III: Continuous Models
- **Chapter 7**: Continuous state estimation
- **Chapter 8**: Active inference in continuous time
- **Chapter 9**: Hierarchical and mixed models

### Part IV: Applications
- **Chapter 10**: Perception and attention
- **Chapter 11**: Motor control and decision making
- **Chapter 12**: Social cognition and communication
- **Chapter 13**: Psychopathology and computational psychiatry

## Impact and Applications

### For Researchers
- **Standard Reference**: Definitive treatment of active inference theory
- **Implementation Guide**: Sufficient detail for building models
- **Cross-Disciplinary**: Bridges neuroscience, AI, and philosophy

### For Students
- **Entry Point**: Accessible introduction to the framework
- **Mathematical Foundations**: Self-contained probability theory review
- **Exercises**: Chapter exercises for self-study

### For Practitioners
- **Model Building**: How to construct generative models for specific problems
- **Software Connection**: Links to SPM and other toolboxes
- **Empirical Applications**: Fitting models to behavioral and neural data

## Related Work

### Foundational Papers
- [[friston_2006]] - Original FEP formulation
- [[friston_2010]] - Unified brain theory review
- [[friston_2017_curiosity]] - Expected free energy

### Tutorials
- [[da_costa_2020]] - Discrete state-space tutorial
- [[smith_2022]] - Step-by-step tutorial
- [[buckley_2017]] - Mathematical review

### Philosophical Context
- [[hohwy_2013]] - The Predictive Mind
- [[clark_2013]] - Predictive brains review
- [[andrews_2021]] - Philosophical critique

## Citations and Influence
As the first comprehensive textbook on active inference, this book has become the standard reference for the field. It consolidates over fifteen years of development in the free energy principle literature into a single, coherent treatment. It is essential reading for anyone working in active inference, predictive processing, or computational neuroscience.

## Reading Guide
1. **Part I** for theoretical foundations
2. **Part II** for discrete implementations (most common in applications)
3. **Part III** for continuous formulations (neuroscience-oriented)
4. **Part IV** for domain-specific applications

---

> **Definitive Textbook**: The authoritative, comprehensive treatment of active inference and the free energy principle.

---

> **Pedagogical**: Builds the theory from first principles with worked examples and consistent notation.

---

> **Practical**: Provides sufficient detail for researchers to build and implement active inference models.
