---
title: "Working Memory, Attention, and Salience in Active Inference"
authors:
  - "Thomas Parr"
  - "Karl J. Friston"
type: citation
status: verified
created: 2025-01-01
year: 2017
journal: "Scientific Reports"
volume: 7
issue: 1
pages: 14678
doi: "10.1038/s41598-017-15249-0"
tags:
  - active_inference
  - working_memory
  - attention
  - salience
  - precision
semantic_relations:
  - type: foundational_for
    links:
      - attention
      - [[knowledge_base/cognitive/working_memory]]
  - type: extends
    links:
      - [[friston_2017_curiosity]]
      - [[friston_2010]]
  - type: cited_by
    links:
      - [[parr_pezzulo_friston_2022]]
      - [[da_costa_2020]]
---

# Working Memory, Attention, and Salience in Active Inference

## Authors
- **Thomas Parr** (UCL)
- **Karl J. Friston** (UCL)

## Publication Details
- **Journal**: Scientific Reports
- **Year**: 2017
- **Volume**: 7
- **Issue**: 1
- **Pages**: 14678
- **DOI**: [10.1038/s41598-017-15249-0](https://doi.org/10.1038/s41598-017-15249-0)

## Abstract
This paper develops active inference accounts of working memory, attention, and salience. It shows how these cognitive functions emerge naturally from the architecture of active inference models without requiring separate, dedicated mechanisms. Working memory is cast as the maintenance of posterior beliefs about hidden states, attention as precision optimization over prediction errors, and salience as the epistemic value of future observations. The paper includes simulations of visual search demonstrating these mechanisms in action.

## Key Contributions

### Working Memory as Belief Maintenance
- **Posterior Beliefs**: Working memory is maintaining beliefs about hidden states
- **Temporal Depth**: Deep temporal models sustain beliefs across time
- **Capacity Limits**: Limited by the complexity of the generative model
- **No Separate Buffer**: Working memory is intrinsic to inference, not a dedicated store

### Attention as Precision
- **Precision Weighting**: Attention is the optimization of precision on prediction errors
- **Gain Modulation**: Neural gain implements precision in cortical circuits
- **Selective Attention**: High precision at attended locations, low elsewhere
- **Attentional Control**: Top-down precision expectations guide attention

### Salience as Epistemic Value
- **Epistemic Affordance**: Salient locations offer high information gain
- **Visual Search**: Saccades directed to resolve uncertainty
- **Bottom-Up Salience**: Surprising stimuli attract attention via prediction error
- **Top-Down Salience**: Task-relevant locations weighted by epistemic value

## Core Concepts

### Active Inference Framework for Cognition
Working memory, attention, and salience are all consequences of minimizing expected free energy:

```
G(pi) = E_q[ln q(s|pi) - ln p(o, s|pi)]
       = Pragmatic Value + Epistemic Value
```

- **Working Memory**: Encoded in `q(s|pi)` -- beliefs about states maintained over time
- **Attention**: Encoded in precision parameters -- gain on prediction errors
- **Salience**: Encoded in epistemic value -- information gain from observations

### Precision and Attention
Attention is modeled as precision optimization:
```
omega* = argmax_omega E_q[-F(omega)]
```

Where omega are precision parameters that modulate the gain on prediction errors at different levels and modalities.

### Visual Search Simulation
The paper demonstrates a visual search model:
1. Agent maintains beliefs about scene content (working memory)
2. Saccades are directed to locations with high epistemic value (salience)
3. Precision is allocated to the currently fixated location (attention)
4. Beliefs are updated after each saccade (inference)

## Neuroscience Connections

### Neural Correlates
- **Prefrontal Cortex**: Working memory as sustained posterior beliefs
- **Parietal Cortex**: Salience maps and attentional control
- **Pulvinar/Thalamus**: Precision modulation and gain control
- **Superior Colliculus**: Saccade generation driven by epistemic value

### Neuromodulation
- **Acetylcholine**: Boosts precision of sensory prediction errors (attention)
- **Dopamine**: Encodes precision of policy selection
- **Norepinephrine**: Global precision modulation (arousal)
- **Serotonin**: Temporal discounting of expected free energy

## Impact and Applications

### Cognitive Neuroscience
- **Unified Cognitive Architecture**: Single framework for multiple cognitive functions
- **Computational Psychiatry**: Aberrant precision in ADHD, schizophrenia
- **Visual Cognition**: Model of saccadic search and scene understanding

### Artificial Intelligence
- **Attention Mechanisms**: Principled attention for AI systems
- **Active Vision**: Visual search strategies for robots
- **Working Memory Networks**: Neural network architectures for memory

## Related Work

### Foundational Papers
- [[friston_2010]] - Free energy principle review
- [[friston_2017_curiosity]] - Expected free energy and epistemic value

### Extensions
- [[parr_pezzulo_friston_2022]] - Comprehensive textbook treatment
- [[hesp_2021]] - Deep active inference and affect

### Related Concepts
- [[da_costa_2020]] - Discrete active inference tutorial
- [[smith_2022]] - Step-by-step tutorial

## Citations and Influence
This paper provided influential demonstrations of how working memory, attention, and salience arise naturally within the active inference framework. It helped establish active inference as a framework for cognitive psychology and not just theoretical neuroscience.

## Reading Guide
1. **Introduction**: Cognitive functions as inference
2. **Working Memory**: Belief maintenance in deep models
3. **Attention**: Precision optimization
4. **Salience**: Epistemic value and visual search
5. **Simulations**: Visual search demonstrations

---

> **Unified Cognition**: Shows how working memory, attention, and salience emerge from the same active inference framework.

---

> **Precision as Attention**: Provides the definitive account of attention as precision weighting of prediction errors.

---

> **Salience as Information**: Redefines salience as epistemic value -- the expected information gain from observations.
