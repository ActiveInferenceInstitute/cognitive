---
title: "Predictive Coding under the Free Energy Principle"
authors:
  - "Karl J. Friston"
type: citation
status: verified
created: 2025-01-01
year: 2009
journal: "Philosophical Transactions of the Royal Society B"
volume: 364
issue: 1521
pages: 1211-1221
doi: "10.1098/rstb.2008.0300"
tags:
  - predictive_coding
  - free_energy
  - neuroscience
  - perception
semantic_relations:
  - type: foundational_for
    links:
      - [[../cognitive/predictive_coding]]
      - [[../cognitive/free_energy_principle]]
  - type: extends
    links:
      - [[../mathematics/free_energy_principle]]
  - type: cited_by
    links:
      - [[friston_2017]]
      - [[friston_2010]]
---

# Predictive Coding under the Free Energy Principle

## Authors
- **Karl J. Friston** (Wellcome Trust Centre for Neuroimaging, University College London)

## Publication Details
- **Journal**: Philosophical Transactions of the Royal Society B
- **Year**: 2009
- **Volume**: 364
- **Issue**: 1521
- **Pages**: 1211-1221
- **DOI**: [10.1098/rstb.2008.0300](https://doi.org/10.1098/rstb.2008.0300)

## Abstract
This paper presents predictive coding as a neurobiologically plausible mechanism for implementing the free energy principle in the brain. The author shows how hierarchical predictive coding minimizes variational free energy through Bayesian inference in the sensory cortex.

## Key Contributions

### Predictive Coding Framework
- **Hierarchical Organization**: Multiple cortical levels for prediction and error
- **Bayesian Inference**: Probabilistic beliefs about sensory causes
- **Error Minimization**: Prediction errors drive belief updating
- **Attentional Mechanisms**: Precision weighting of prediction errors

### Free Energy Principle Implementation
- **Variational Free Energy**: Information-theoretic bound on surprise
- **Perception as Inference**: Sensory input constrains posterior beliefs
- **Active Inference**: Action minimizes expected free energy
- **Learning**: Synaptic plasticity implements parameter optimization

### Neurobiological Mechanisms
- **Forward Connections**: Predictions from higher to lower areas
- **Backward Connections**: Prediction errors from lower to higher areas
- **Lateral Connections**: Horizontal prediction and error propagation
- **Neuromodulation**: Precision control through neuromodulatory systems

## Core Concepts

### Predictive Coding
Predictive coding proposes that the brain maintains a hierarchical model of the sensory world and continuously predicts sensory input at multiple levels. When predictions fail, prediction errors are propagated backward to update higher-level representations.

### Variational Free Energy
The variational free energy provides a bound on the surprise (negative log evidence) of sensory data:
```
F = E_{q} [ln q(μ) - ln p(y,μ|m)]
```

Where:
- `q(μ)`: Approximate posterior over hidden causes
- `p(y,μ|m)`: Generative model likelihood
- `m`: Model parameters

### Hierarchical Message Passing
Information flows bidirectionally in cortical hierarchies:
- **Bottom-up**: Prediction errors signal mismatches
- **Top-down**: Predictions constrain lower-level representations
- **Lateral**: Contextual information modulates processing

## Mathematical Formalism

### Generative Model
The generative model assumes sensory data `y` arise from hidden causes `μ`:
```
p(y|μ,m) = N(y; g(μ), Π⁻¹)
```

Where:
- `g(μ)`: Generative function mapping causes to predictions
- `Π`: Precision (inverse variance) of sensory noise

### Variational Inference
Perception minimizes variational free energy through gradient descent:
```
∂F/∂μ = ε = y - g(μ)
```

Where `ε` represents prediction error.

### Learning
Synaptic plasticity implements learning of model parameters:
```
∂F/∂m ∝ <∂ε/∂m>
```

## Neurobiological Evidence

### Visual Cortex
- **V1**: Primary visual cortex implements prediction error units
- **V2/V4**: Higher visual areas represent increasingly abstract features
- **IT Cortex**: Object representations at highest levels

### Functional Anatomy
- **Forward Model**: Motor cortex predicts sensory consequences
- **Inverse Model**: Parietal cortex infers causes from sensory input
- **Cerebellum**: State estimation and prediction

### Neuromodulatory Systems
- **Acetylcholine**: Increases precision of attended stimuli
- **Dopamine**: Encodes precision of reward predictions
- **Norepinephrine**: Global precision modulation

## Impact and Applications

### Neuroscience
- **Perception**: Hierarchical inference in sensory systems
- **Attention**: Precision weighting mechanisms
- **Motor Control**: Forward and inverse models
- **Consciousness**: Self-evidencing through prediction

### Computational Models
- **Neural Networks**: Predictive coding implementations
- **Reinforcement Learning**: Free energy as intrinsic reward
- **Computer Vision**: Hierarchical feature learning

### Clinical Applications
- **Hallucinations**: Aberrant precision weighting
- **Delusions**: False priors in generative models
- **Autism**: Altered precision in social prediction

## Related Work

### Foundational Papers
- [[friston_2005]] - Dynamic causal modeling
- [[friston_2007]] - Variational filtering
- [[friston_2010]] - Free energy principle

### Extensions
- [[friston_2017]] - Active Inference process theory
- [[clark_2013]] - Predictive brains review
- [[hodgkinson_2019]] - Motor control applications

## Code and Implementations
- **SPM**: Statistical Parametric Mapping (MATLAB)
- **Neural Networks**: Various deep learning implementations
- **Active Inference**: Modern implementations

## Citations and Influence
This paper has been cited over 800 times and established predictive coding as the canonical implementation of the free energy principle in neuroscience. It provides the computational foundation for understanding hierarchical perception and learning in the brain.

## Reading Guide
1. **Introduction**: Free energy principle overview
2. **Predictive Coding**: Hierarchical inference mechanism
3. **Mathematical Formalism**: Variational free energy derivation
4. **Neurobiological Evidence**: Brain implementation
5. **Applications**: Extensions to behavior and pathology

---

> **Canonical Paper**: Establishes predictive coding as the neurobiological implementation of the free energy principle.

---

> **Hierarchical Inference**: Demonstrates how the brain performs Bayesian inference through hierarchical message passing.

---

> **Computational Neuroscience**: Bridges theoretical neuroscience with practical implementations in neural systems.

---

> **Foundation for Active Inference**: Provides the perceptual mechanism that Active Inference extends to action and learning.
