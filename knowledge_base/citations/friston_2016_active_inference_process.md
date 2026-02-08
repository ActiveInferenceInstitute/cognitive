---
title: "Active Inference and Learning"
authors:
  - "Karl J. Friston"
  - "Thomas FitzGerald"
  - "Francesco Rigoli"
  - "Philipp Schwartenbeck"
  - "Giovanni Pezzulo"
type: citation
status: verified
created: 2025-01-01
year: 2016
journal: "Neuroscience & Biobehavioral Reviews"
volume: 68
pages: 862-879
doi: "10.1016/j.neubiorev.2016.06.022"
tags:
  - active_inference
  - learning
  - parameter_estimation
  - structure_learning
  - bayesian_learning
semantic_relations:
  - type: foundational_for
    links:
      - learning
      - bayesian learning
  - type: extends
    links:
      - [[friston_2010]]
      - [[friston_2009]]
  - type: cited_by
    links:
      - [[friston_2017_curiosity]]
      - [[da_costa_2020]]
      - [[parr_pezzulo_friston_2022]]
---

# Active Inference and Learning

## Authors
- **Karl J. Friston** (UCL)
- **Thomas FitzGerald** (UCL)
- **Francesco Rigoli** (UCL)
- **Philipp Schwartenbeck** (UCL/Oxford)
- **Giovanni Pezzulo** (ISTC-CNR)

## Publication Details
- **Journal**: Neuroscience & Biobehavioral Reviews
- **Year**: 2016
- **Volume**: 68
- **Pages**: 862-879
- **DOI**: [10.1016/j.neubiorev.2016.06.022](https://doi.org/10.1016/j.neubiorev.2016.06.022)

## Abstract
This paper provides a comprehensive treatment of learning within the active inference framework. The authors show how different forms of learning -- perceptual learning, associative learning, structure learning, and habit formation -- all emerge from the same imperative to minimize variational free energy. The paper distinguishes between inference (state estimation at fast timescales), learning (parameter estimation at medium timescales), and model selection (structure learning at slow timescales), showing how these nested optimizations are all instances of free energy minimization.

## Key Contributions

### Unified Learning Framework
- **State Estimation**: Inference about hidden states (fast -- perception)
- **Parameter Learning**: Updating model parameters (medium -- learning)
- **Structure Learning**: Changing model structure (slow -- development)
- **Policy Learning**: Forming habits (very slow -- automatization)

### Timescales of Optimization
- **Milliseconds**: Perceptual inference (state estimation)
- **Seconds-Minutes**: Short-term learning (parameter updates)
- **Hours-Days**: Consolidation and generalization
- **Weeks-Months**: Structural changes to generative models
- **Years**: Fundamental model architecture changes

### Bayesian Learning Mechanisms
- **Dirichlet Learning**: Concentration parameter updates for categorical distributions
- **Precision Learning**: Updating confidence in model components
- **Novelty Detection**: Identifying when model structure is inadequate
- **Forgetting**: Controlled loss of parameters to prevent overfitting

### Neural Implementation
- **Hebbian Plasticity**: Correlation-based learning as free energy minimization
- **Synaptic Consolidation**: Parameter stabilization through precision
- **Neuromodulation**: Learning rates controlled by neuromodulatory systems
- **Sleep**: Model consolidation during offline processing

## Core Concepts

### Nested Optimization
All forms of learning minimize the same free energy at different timescales:

```
Fast (ms):    dmu/dt = -dF/dmu        # State estimation (perception)
Medium (s):   dtheta/dt = -dF/dtheta  # Parameter learning
Slow (h/d):   dm/dt = -dF/dm          # Structure learning
Very slow:    dE/dt = -dF/dE          # Habit formation
```

### Parameter Learning (Dirichlet Updates)
For discrete models, learning updates concentration parameters:
```
a_posterior = a_prior + observations
```

Where:
- `a_prior`: Prior concentration parameters (before experience)
- `observations`: Sufficient statistics of experienced data
- `a_posterior`: Updated parameters (after experience)

The effective learning rate is:
```
eta = 1 / sum(a_prior)  # Learning rate decreases as evidence accumulates
```

### Structure Learning
Bayesian model comparison selects model structure:
```
m* = argmin_m F(m)
```

Where F(m) is the free energy for model m, balancing accuracy (fit) and complexity (parsimony).

### Habit Formation
Habits emerge from learning about policy distributions:
```
E_posterior = E_prior + N(pi)  # Policy prior updated by experience
```

Where `N(pi)` counts how often policy pi was selected. Over time, frequently selected policies become habitual.

## Mathematical Formalism

### Free Energy for Learning
```
F = E_q[ln q(s, theta) - ln p(o, s, theta|m)]
  = E_q[ln q(s) - ln p(o|s, theta)] + KL[q(theta) || p(theta|m)]
  = Inference_cost + Learning_cost
```

### Learning Rate Dynamics
Natural learning rate from Dirichlet:
```
eta(t) = 1 / (N_0 + t)
```

Where `N_0` is the effective prior sample size. This naturally implements:
- Fast learning early (high eta)
- Slow learning later (low eta)
- Catastrophic forgetting prevention

### Precision of Parameters
Confidence in learned parameters:
```
Pi_theta = d^2F/dtheta^2  # Fisher information
```

High precision = confident in parameters = slow learning.
Low precision = uncertain about parameters = fast learning.

## Neuroscience Connections

### Neural Correlates of Learning
- **Hippocampus**: Episodic learning and memory consolidation
- **Basal Ganglia**: Habit formation and policy learning
- **Prefrontal Cortex**: Structure learning and model selection
- **Cerebellum**: Fine-grained parameter optimization

### Neuromodulation and Learning Rates
- **Dopamine**: Precision of reward predictions (learning rate for action values)
- **Acetylcholine**: Precision of sensory predictions (learning rate for perception)
- **Norepinephrine**: Global learning rate modulation (uncertainty signal)
- **Serotonin**: Temporal discounting and patience in learning

### Sleep and Consolidation
- **NREM Sleep**: Model consolidation and parameter stabilization
- **REM Sleep**: Structure learning and model testing
- **Replay**: Hippocampal replay as offline parameter optimization
- **Pruning**: Synaptic downscaling as complexity reduction

## Impact and Applications

### Cognitive Science
- **Learning Theory**: Unified account of multiple learning types
- **Memory**: Different memory systems as different timescales of optimization
- **Development**: Cognitive development as structure learning
- **Education**: Implications for teaching and training

### Artificial Intelligence
- **Meta-Learning**: Learning to learn through nested optimization
- **Continual Learning**: Preventing catastrophic forgetting
- **Architecture Search**: Structure learning as neural architecture search
- **Curriculum Learning**: Optimal ordering of learning experiences

### Computational Psychiatry
- **Autism**: Aberrant precision of parameters (too fast/slow learning)
- **PTSD**: Failed structure learning after trauma
- **Addiction**: Aberrant habit formation
- **OCD**: Excessive model checking and uncertainty

## Related Work

### Foundational Papers
- [[friston_2010]] - Free energy principle review
- [[friston_2009]] - Predictive coding under FEP

### Extensions
- [[friston_2017_curiosity]] - Epistemic value and curiosity
- [[da_costa_2020]] - Discrete active inference with learning
- [[parr_pezzulo_friston_2022]] - Textbook treatment

### Related Concepts
- [[smith_2022]] - Empirical tutorial on model fitting
- [[pezzulo_2015]] - Adaptive behavioral control

## Citations and Influence
This paper provides the definitive treatment of learning within the active inference framework. It has been widely cited for its clear articulation of how different forms of learning emerge from nested timescales of free energy minimization and for connecting this framework to neural mechanisms of learning and memory.

## Reading Guide
1. **Introduction**: Learning in the brain
2. **Timescales**: Nested optimization at different timescales
3. **Parameter Learning**: Bayesian approaches to learning
4. **Structure Learning**: Model selection and development
5. **Habit Formation**: Automatization through policy priors
6. **Neural Mechanisms**: How the brain implements these forms of learning

---

> **Unified Learning**: Shows how perception, learning, structure learning, and habit formation are all nested timescales of free energy minimization.

---

> **Timescale Hierarchy**: Articulates the clear timescale hierarchy from millisecond inference to developmental structure learning.

---

> **Neural Implementation**: Connects each form of learning to specific neural mechanisms and neuromodulatory systems.
