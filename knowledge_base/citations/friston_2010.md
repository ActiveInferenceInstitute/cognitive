---
title: "The Free-Energy Principle: A Unified Brain Theory?"
authors:
  - "Karl J. Friston"
type: citation
status: verified
created: 2025-01-01
year: 2010
journal: "Nature Reviews Neuroscience"
volume: 11
issue: 2
pages: 127-138
doi: "10.1038/nrn2787"
tags:
  - free_energy
  - brain_theory
  - neuroscience
  - predictive_processing
semantic_relations:
  - type: foundational_for
    links:
      - [[../mathematics/free_energy_principle]]
      - [[../cognitive/predictive_processing]]
  - type: cited_by
    links:
      - [[friston_2017]]
      - [[parr_2019]]
---

# The Free-Energy Principle: A Unified Brain Theory?

## Author
- **Karl J. Friston** (Wellcome Trust Centre for Neuroimaging, University College London)

## Publication Details
- **Journal**: Nature Reviews Neuroscience
- **Year**: 2010
- **Volume**: 11
- **Issue**: 2
- **Pages**: 127-138
- **DOI**: [10.1038/nrn2787](https://doi.org/10.1038/nrn2787)

## Abstract
This review article introduces the free-energy principle as a unified theory of brain function. Friston argues that the brain minimizes variational free energy to maintain accurate internal models of the world, providing a single principle that explains perception, action, and learning.

## Key Contributions

### Free-Energy Principle
The free-energy principle states that biological systems minimize variational free energy to avoid surprising states. This principle provides a unified account of brain function and adaptive behavior.

### Variational Inference
The brain performs approximate Bayesian inference by minimizing variational free energy, which bounds the surprise associated with sensory inputs under a generative model.

### Predictive Coding
Perception involves hierarchical prediction and error correction, where higher cortical areas predict the activity of lower areas, and prediction errors are propagated upward.

### Active Inference
Action serves to minimize expected free energy by sampling sensory data that confirm the brain's predictions, rather than merely responding to external stimuli.

## Theoretical Framework

### Variational Free Energy
```
F = E_{q(θ)}[ln q(θ) - ln p(y,θ|m)]
```

This quantity is minimized through:
- **Perception**: Optimizing beliefs about hidden causes
- **Action**: Selecting policies that minimize expected free energy
- **Learning**: Updating model parameters

### Generative Models
The brain maintains hierarchical generative models that predict sensory inputs. These models embody the brain's understanding of how the world works.

### Markov Blankets
Biological systems are separated from their environment by Markov blankets, which define the interface between internal and external states.

## Neurobiological Evidence

### Predictive Coding
- **Hierarchical Organization**: Cortical hierarchies implement predictive coding
- **Error Units**: Specialized neurons encode prediction errors
- **Precision Estimation**: Neuromodulators control the precision of prediction errors

### Functional Anatomy
- **Extrinsic Connections**: Forward connections carry predictions
- **Intrinsic Connections**: Backward connections carry prediction errors
- **Superficial Pyramidal Cells**: Implement prediction error computation

### Electrophysiological Evidence
- **Mismatch Negativity**: Neural response to prediction errors
- **Repetition Suppression**: Reduced responses to predictable stimuli
- **Sensory Attenuation**: Reduced sensory responses during self-generated actions

## Implications

### Perception
Perception is an active process of hypothesis testing, where the brain continuously predicts sensory inputs and updates beliefs based on prediction errors.

### Action
Actions are selected to minimize expected free energy, providing a principled account of goal-directed behavior and exploration.

### Learning
Learning corresponds to updating generative models to better predict future sensory inputs, unifying different forms of learning under a single principle.

### Consciousness
The free-energy principle may provide insights into consciousness, where conscious states correspond to states of high evidence for the brain's model.

## Critical Assessment

### Strengths
- **Unified Framework**: Single principle explains diverse brain functions
- **Mathematical Precision**: Enables quantitative predictions
- **Biological Plausibility**: Consistent with neuroanatomical and physiological data
- **Broad Applicability**: From single neurons to complex behaviors

### Challenges
- **Computational Complexity**: Approximate inference may be intractable
- **Empirical Validation**: Difficult to test core predictions directly
- **Philosophical Questions**: Relationship to consciousness and subjective experience
- **Alternative Theories**: Competition with other brain theories

## Impact

This paper introduced the free-energy principle to a broad neuroscience audience and established it as a major theoretical framework. It has been cited over 2000 times and continues to influence research in neuroscience, psychology, and artificial intelligence.

## Related Work
- [[friston_2009]] - Predictive coding in visual cortex
- [[friston_2005]] - Dynamic causal modeling
- [[friston_2017]] - Active Inference process theory

---

> **Unified Theory**: This paper presents the free-energy principle as a comprehensive theory of brain function.

---

> **Predictive Brain**: Establishes the brain as a prediction machine that minimizes surprise.

---

> **Foundational Work**: Essential reading for understanding modern theories of brain function.
