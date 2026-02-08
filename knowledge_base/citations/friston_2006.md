---
title: "A Free Energy Principle for the Brain"
authors:
  - "Karl J. Friston"
  - "James Kilner"
  - "Lee Harrison"
type: citation
status: verified
created: 2025-01-01
year: 2006
journal: "Journal of Physiology-Paris"
volume: 100
issue: 1-3
pages: 70-87
doi: "10.1016/j.jphysparis.2006.10.001"
tags:
  - free_energy
  - variational_inference
  - perception
  - neuroscience
  - foundational
semantic_relations:
  - type: foundational_for
    links:
      - [[knowledge_base/mathematics/free_energy_principle]]
      - [[knowledge_base/cognitive/predictive_coding]]
  - type: cited_by
    links:
      - [[friston_2009]]
      - [[friston_2010]]
      - [[buckley_2017]]
---

# A Free Energy Principle for the Brain

## Authors
- **Karl J. Friston** (Wellcome Trust Centre for Neuroimaging, University College London)
- **James Kilner**
- **Lee Harrison**

## Publication Details
- **Journal**: Journal of Physiology-Paris
- **Year**: 2006
- **Volume**: 100
- **Issue**: 1-3
- **Pages**: 70-87
- **DOI**: [10.1016/j.jphysparis.2006.10.001](https://doi.org/10.1016/j.jphysparis.2006.10.001)

## Abstract
This paper introduces the free energy principle as a unified theory of brain function. It proposes that the brain minimizes a variational free energy bound on surprise (negative log-evidence) to maintain homeostasis and perform inference about the causes of sensory input. This is the original formulation of the FEP as applied to neural systems, laying the groundwork for all subsequent developments in active inference and predictive processing.

## Key Contributions

### Original FEP Formulation
- **Variational Free Energy**: Introduced as the quantity the brain minimizes
- **Surprise Minimization**: Biological systems avoid surprising states to persist
- **Bayesian Brain**: The brain as an inference machine implementing approximate Bayesian inference
- **Unification**: Single principle accounts for perception, learning, and action

### Thermodynamic Grounding
- **Entropy Reduction**: Living systems resist the second law by bounding entropy
- **Information-Theoretic**: Free energy as an information-theoretic quantity
- **Helmholtz Machine**: Connection to variational autoencoders and recognition models
- **Biological Plausibility**: Neural process theory for implementation

### Perceptual Inference
- **Recognition Density**: Brain encodes approximate posterior beliefs
- **Generative Models**: Internal models of how sensory data are generated
- **Prediction Error**: Mismatch signals drive belief updating
- **Hierarchical Processing**: Cortical hierarchies implement hierarchical inference

## Core Concepts

### The Free Energy Principle
The central claim is that biological systems minimize variational free energy to maintain their structural and functional integrity. Free energy provides an upper bound on surprise:

```
F >= -ln p(y|m)  (surprise)
F = E_q[ln q(mu) - ln p(y, mu|m)]
```

Where:
- `F`: Variational free energy
- `q(mu)`: Recognition density (approximate posterior)
- `p(y, mu|m)`: Generative model (joint density over observations and causes)
- `m`: Model parameters

### Why Free Energy?
Organisms cannot compute surprise directly (it requires integrating over all hidden causes). Free energy provides a tractable upper bound that can be minimized through local computations in neural circuits.

### Connection to Thermodynamics
The principle draws on Helmholtz's work on free energy in thermodynamics, connecting information-theoretic surprise to physical entropy. Living systems maintain low-entropy states by minimizing free energy.

## Mathematical Formalism

### Variational Free Energy Decomposition
Free energy can be decomposed in two equivalent ways:

**Energy minus entropy:**
```
F = E_q[ln q(mu)] - E_q[ln p(y, mu|m)]
F = -H[q] + E_q[-ln p(y, mu|m)]
```

**Complexity minus accuracy:**
```
F = KL[q(mu) || p(mu|m)] - E_q[ln p(y|mu, m)]
F = Complexity - Accuracy
```

### Gradient Descent on Free Energy
The brain minimizes free energy through gradient descent on sufficient statistics:
```
dmu/dt = -dF/dmu
```

This yields prediction error minimization in neural circuits.

## Neuroscience Evidence

### Neural Implementation
- **Cortical Hierarchies**: Forward and backward connections implement prediction and error
- **Synaptic Plasticity**: Learning minimizes free energy over parameters
- **Neuromodulation**: Precision weighting through ascending neuromodulatory systems
- **Oscillations**: Neural oscillations reflect message passing in hierarchical inference

### Sensory Processing
- **Vision**: Hierarchical visual processing as inference
- **Audition**: Auditory scene analysis as free energy minimization
- **Somatosensation**: Body schema as generative model

## Impact and Applications

### Theoretical Neuroscience
- **Unified Brain Theory**: Single principle for perception, action, and learning
- **Computational Psychiatry**: Aberrant inference as basis for mental illness
- **Consciousness Studies**: Self-evidencing as basis for phenomenal experience

### Machine Learning
- **Variational Autoencoders**: Deep connections to generative modeling
- **Bayesian Deep Learning**: Free energy as objective function
- **Active Learning**: Epistemic value and curiosity

## Related Work

### Precursors
- Helmholtz (1867) - Free energy in thermodynamics
- Dayan et al. (1995) - Helmholtz machine
- Rao & Ballard (1999) - Predictive coding

### Extensions
- [[friston_2009]] - Predictive coding under the FEP
- [[friston_2010]] - Unified brain theory review
- [[buckley_2017]] - Mathematical review of the FEP

## Citations and Influence
This is the founding paper of the free energy principle as applied to neuroscience. It has been cited thousands of times and spawned an entire research program encompassing active inference, predictive processing, and computational psychiatry. Every subsequent development in the FEP literature builds on the framework introduced here.

## Reading Guide
1. **Introduction**: Motivation from thermodynamics and information theory
2. **Free Energy**: Mathematical definition and properties
3. **Neural Implementation**: How the brain could minimize free energy
4. **Perception**: Inference through prediction error minimization
5. **Implications**: Unifying perception, action, and learning

---

> **Founding Paper**: The original formulation of the free energy principle for biological systems and the brain.

---

> **Variational Inference**: Introduces the brain as performing approximate Bayesian inference by minimizing variational free energy.

---

> **Unified Theory**: Proposes a single principle to account for perception, learning, action, and homeostasis.
