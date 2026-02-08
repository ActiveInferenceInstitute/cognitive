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
  - unified_theory
  - perception
  - action
  - learning
  - neuroscience
semantic_relations:
  - type: foundational_for
    links:
      - [[knowledge_base/mathematics/free_energy_principle]]
      - [[knowledge_base/cognitive/active_inference]]
      - [[knowledge_base/cognitive/predictive_coding]]
  - type: extends
    links:
      - [[friston_2006]]
      - [[friston_2009]]
  - type: cited_by
    links:
      - [[friston_2012]]
      - [[friston_2013]]
      - [[clark_2013]]
      - [[hohwy_2013]]
---

# The Free-Energy Principle: A Unified Brain Theory?

## Authors
- **Karl J. Friston** (Wellcome Trust Centre for Neuroimaging, University College London)

## Publication Details
- **Journal**: Nature Reviews Neuroscience
- **Year**: 2010
- **Volume**: 11
- **Issue**: 2
- **Pages**: 127-138
- **DOI**: [10.1038/nrn2787](https://doi.org/10.1038/nrn2787)

## Abstract
This landmark review paper asks whether the free energy principle can serve as a unified theory of the brain. Friston argues that the imperative to minimize variational free energy -- an upper bound on surprise -- unifies perception, action, and learning under a single theoretical framework. The paper reviews evidence from neuroscience, psychology, and computational modeling to support this claim, and connects the FEP to established theories including predictive coding, Bayesian brain, and optimal control.

## Key Contributions

### Unification of Brain Functions
- **Perception**: Optimizing internal representations (recognition density)
- **Action**: Changing sensory input to match predictions (active inference)
- **Attention**: Optimizing precision of prediction errors
- **Learning**: Optimizing model parameters over longer timescales
- **Sleep/Development**: Optimizing model structure

### Connections to Existing Theories
- **Bayesian Brain Hypothesis**: FEP as implementation of Bayesian inference
- **Predictive Coding**: Neural process theory for free energy minimization
- **Optimal Control Theory**: Action as fulfilling predictions, not separate optimization
- **Information Theory**: Surprise and entropy as core quantities
- **Helmholtz Machine**: Historical precursor in machine learning

### Two Ways to Minimize Free Energy
- **Perceptual Inference**: Change internal states to better predict sensory input
- **Active Inference**: Change the world to match internal predictions

## Core Concepts

### The Free Energy Principle
Any self-organizing system at equilibrium with its environment must minimize free energy. For biological agents:

```
F = -ln p(y|m) + KL[q(mu) || p(mu|y,m)]
F = Surprise + KL Divergence
```

Since KL divergence is non-negative, minimizing F minimizes an upper bound on surprise.

### Active Inference
Action is recast not as optimizing a value function (as in reinforcement learning) but as minimizing prediction error by acting on the world:

```
a = argmin_a F(y(a), mu)
```

The agent acts to make its sensory input conform to its predictions, unifying perception and action under a single imperative.

### Hierarchical Predictive Coding
The neural implementation proceeds through hierarchical message passing:
- **Top-down predictions**: Higher areas predict activity in lower areas
- **Bottom-up prediction errors**: Lower areas signal mismatches
- **Lateral connections**: Encode precision (confidence) of predictions

### Attention as Precision Optimization
Attention is reconceived as the optimization of precision -- the confidence placed on prediction errors at different levels and modalities:

```
pi = exp(gamma)  # precision as gain on prediction errors
```

## Mathematical Formalism

### Generative Model
The brain maintains a hierarchical generative model:
```
p(y, mu_1, ..., mu_n | m) = p(y|mu_1) * prod_i p(mu_i|mu_{i+1})
```

### Free Energy Minimization
Perception, action, and learning all follow from minimizing the same quantity:
- **Perception**: dmu/dt = -dF/dmu (state estimation)
- **Action**: da/dt = -dF/da (active inference)
- **Learning**: dtheta/dt = -dF/dtheta (parameter optimization)
- **Model selection**: argmin_m F (structure learning)

## Neuroscience Evidence

### Perception
- **Predictive coding in visual cortex**: Hierarchical prediction and error
- **Mismatch negativity**: Prediction error signals in auditory cortex
- **Perceptual illusions**: Explained by strong priors overriding sensory evidence

### Action
- **Motor control**: Spinal reflex arcs as prediction error minimizers
- **Goal-directed behavior**: Proprioceptive predictions drive movement
- **Mirror neurons**: Prediction of others' actions

### Learning
- **Hebbian plasticity**: Correlational learning minimizes free energy
- **Reward learning**: Dopamine as precision signal for reward predictions
- **Structural plasticity**: Pruning as model optimization

### Attention
- **Cholinergic modulation**: Acetylcholine as precision modulator
- **Gain control**: Neural gain as precision implementation
- **Biased competition**: Selection through precision weighting

## Impact and Applications

### Neuroscience
- **Computational Psychiatry**: Mental illness as aberrant inference
- **Neuroimaging**: Dynamic causal modeling of brain connectivity
- **Developmental Neuroscience**: Brain development as model optimization

### Cognitive Science
- **Embodied Cognition**: Action as integral to cognition
- **Situated Cognition**: Environment as part of the generative model
- **4E Cognition**: Embedded, embodied, enacted, extended

### Artificial Intelligence
- **Generative Models**: Deep generative models for perception
- **Active Inference Agents**: Autonomous agents minimizing free energy
- **Curiosity-Driven Learning**: Epistemic value from free energy

## Related Work

### Foundational Papers
- [[friston_2006]] - Original FEP formulation
- [[friston_2009]] - Predictive coding under the FEP

### Reviews and Extensions
- [[clark_2013]] - Predictive brains review
- [[hohwy_2013]] - The Predictive Mind (book)
- [[buckley_2017]] - Mathematical review

### Applications
- [[friston_2012]] - Temporal aspects
- [[friston_2013]] - Life as we know it
- [[parr_friston_2017]] - Working memory and attention

## Citations and Influence
This is arguably the most influential paper in the FEP literature, published in one of neuroscience's premier review journals. With thousands of citations, it established the free energy principle as a serious candidate for a unified brain theory and catalyzed the fields of active inference and computational psychiatry. It remains the standard entry point for neuroscientists encountering the FEP.

## Reading Guide
1. **Box 1**: Summary of the free energy principle
2. **Perception**: Predictive coding as neural process theory
3. **Action**: Active inference unifying perception and action
4. **Learning**: Parameter and structure optimization
5. **Figure 3**: Hierarchical message passing diagram (key visual)

---

> **Landmark Review**: The most cited and influential overview of the free energy principle, published in Nature Reviews Neuroscience.

---

> **Unification**: Demonstrates how perception, action, attention, and learning all follow from a single imperative to minimize free energy.

---

> **Active Inference**: Introduces the radical idea that action serves to fulfill predictions rather than maximize reward.
