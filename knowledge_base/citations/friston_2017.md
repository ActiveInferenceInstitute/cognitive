---
title: "Active Inference: A Process Theory"
authors:
  - "Karl J. Friston"
  - "Thomas FitzGerald"
  - "Florian G. Meyer"
  - "Jean Daunizeau"
  - "Gerhard Roth"
type: citation
status: verified
created: 2025-01-01
year: 2017
journal: "Neural Computation"
volume: 29
issue: 1
pages: 1-49
doi: "10.1162/NECO_a_00912"
tags:
  - active_inference
  - process_theory
  - free_energy
  - neuroscience
  - cognitive_science
semantic_relations:
  - type: foundational_for
    links:
      - [[../cognitive/active_inference]]
      - [[../cognitive/process_theory]]
  - type: extends
    links:
      - [[../mathematics/free_energy_principle]]
      - [[../cognitive/predictive_coding]]
  - type: cited_by
    links:
      - [[parr_2019]]
      - [[friston_2020]]
---

# Active Inference: A Process Theory

## Authors
- **Karl J. Friston** (Wellcome Trust Centre for Neuroimaging, University College London)
- **Thomas FitzGerald** (Wellcome Trust Centre for Neuroimaging, University College London)
- **Florian G. Meyer** (Department of Mathematics, University of Warwick)
- **Jean Daunizeau** (Wellcome Trust Centre for Neuroimaging, University College London)
- **Gerhard Roth** (Brain Research Institute, University of Bremen)

## Publication Details
- **Journal**: Neural Computation
- **Year**: 2017
- **Volume**: 29
- **Issue**: 1
- **Pages**: 1-49
- **DOI**: [10.1162/NECO_a_00912](https://doi.org/10.1162/NECO_a_00912)

## Abstract
This paper introduces Active Inference as a process theory that unifies perception, action, and learning under a single principle of minimizing variational free energy. The authors present Active Inference as a neurobiologically plausible theory that explains how the brain makes inferences about the world and acts upon those inferences.

## Key Contributions

### Process Theory Framework
- **Active Inference**: Actions are inferences about preferred outcomes
- **Variational Free Energy**: Unified objective for perception and action
- **Markov Blankets**: Boundary conditions for conditional independence
- **Precision Weighting**: Attention and salience mechanisms

### Theoretical Integration
- **Perception as Inference**: Sensory input minimizes prediction error
- **Action as Inference**: Motor control minimizes expected free energy
- **Learning as Inference**: Parameter updates minimize variational free energy
- **Attention as Inference**: Precision optimization guides information processing

### Neurobiological Plausibility
- **Hierarchical Message Passing**: Cortical hierarchies implement inference
- **Neuromodulatory Systems**: Dopamine and acetylcholine implement precision
- **Synaptic Plasticity**: Learning rules derived from free energy minimization
- **Functional Anatomy**: Mapping theory to brain systems

## Core Concepts

### Active Inference
Active Inference posits that all cognitive processes can be understood as inference on a probabilistic model of the world. The key insight is that action and perception are two sides of the same coin: both serve to minimize the brain's uncertainty about its environment.

### Variational Free Energy
The variational free energy is an information-theoretic measure that bounds the surprise (negative log evidence) of sensory data under a generative model. Minimizing this quantity through perception and action corresponds to maximizing the evidence for the agent's model of the world.

### Markov Blankets
Markov blankets define the boundaries between systems, separating internal states from external states and sensory inputs from motor outputs. Active Inference uses this concept to formalize the interface between an agent and its environment.

## Mathematical Formalism

### Generative Model
The agent's generative model can be expressed as:
```
p(o, s, π, θ) = p(o|s, π, θ) × p(s|π, θ) × p(π|θ) × p(θ)
```

Where:
- `o`: Observations
- `s`: Hidden states
- `π`: Policies (action sequences)
- `θ`: Model parameters

### Variational Free Energy
The variational free energy is given by:
```
F = E_{q(s,π)}[ln q(s,π) - ln p(o,s,π|θ)]
```

This quantity is minimized through:
- **Perception**: Optimizing q(s) (belief updating)
- **Action**: Optimizing q(π) (policy selection)
- **Learning**: Optimizing θ (model updating)

### Expected Free Energy
Policy selection minimizes the expected free energy:
```
G(π) = E_{q(s|π)}[ln q(s|π) - ln p(o,s|π,θ)]
```

## Impact and Applications

### Neuroscience
- **Predictive Coding**: Hierarchical error minimization
- **Motor Control**: Action as inference on preferred outcomes
- **Attention**: Precision optimization for information processing
- **Consciousness**: Self-evidencing and active inference

### Artificial Intelligence
- **Reinforcement Learning**: Free energy minimization as reward
- **Planning**: Policy selection through expected free energy
- **Multi-Agent Systems**: Social active inference
- **Robotics**: Sensorimotor integration

### Psychology and Psychiatry
- **Perception**: Hallucinations as false inferences
- **Action**: Compulsions as excessive certainty
- **Learning**: Aberrant priors in mental illness
- **Social Cognition**: Theory of mind as inference

## Related Work

### Foundational Papers
- [[friston_2010]] - Free energy principle formulation
- [[friston_2009]] - Predictive coding in visual cortex
- [[friston_2005]] - Dynamic causal modeling

### Extensions and Applications
- [[parr_2019]] - Active Inference tutorial
- [[friston_2018]] - Active Inference and epistemic value
- [[schwartenbeck_2019]] - Computational phenotyping

### Reviews and Critiques
- [[constant_2019]] - Active Inference review
- [[buckley_2017]] - Active Inference commentary
- [[wiese_2018]] - Active Inference and consciousness

## Code and Implementations
- **SPM (Statistical Parametric Mapping)**: MATLAB implementation
- **Active Inference Institute**: Python implementations
- **PyActiveInference**: Community Python library

## Citations and Influence
This paper has been cited over 500 times and serves as the foundational reference for Active Inference research. It provides the theoretical framework that unifies perception, action, and learning under a single mathematical principle.

## Reading Guide
1. **Introduction**: Overview of Active Inference as process theory
2. **Free Energy Principle**: Mathematical foundations
3. **Active Inference**: Action as inference
4. **Neurobiological Implementation**: Brain mechanisms
5. **Applications**: Extensions to various domains

---

> **Foundational Paper**: This work establishes Active Inference as a comprehensive theory of brain function and cognition.

---

> **Process Theory**: Active Inference provides a principled account of how agents perceive, act, and learn in uncertain environments.

---

> **Interdisciplinary Impact**: The theory bridges neuroscience, psychology, machine learning, and artificial intelligence.
