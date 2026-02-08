---
title: "Active Inference, Homeostatic Regulation and Adaptive Behavioural Control"
authors:
  - "Giovanni Pezzulo"
  - "Francesco Rigoli"
  - "Karl J. Friston"
type: citation
status: verified
created: 2025-01-01
year: 2015
journal: "Progress in Neurobiology"
volume: 134
pages: 17-35
doi: "10.1016/j.pneurobio.2015.09.001"
tags:
  - active_inference
  - homeostasis
  - allostasis
  - adaptive_behavior
  - interoception
semantic_relations:
  - type: foundational_for
    links:
      - [[knowledge_base/biology/homeostasis]]
      - adaptive behavior
  - type: extends
    links:
      - [[friston_2010]]
      - [[friston_2013]]
  - type: cited_by
    links:
      - [[seth_2021]]
      - [[hesp_2021]]
      - [[parr_pezzulo_friston_2022]]
---

# Active Inference, Homeostatic Regulation and Adaptive Behavioural Control

## Authors
- **Giovanni Pezzulo** (Institute of Cognitive Sciences and Technologies, CNR)
- **Francesco Rigoli** (UCL)
- **Karl J. Friston** (UCL)

## Publication Details
- **Journal**: Progress in Neurobiology
- **Year**: 2015
- **Volume**: 134
- **Pages**: 17-35
- **DOI**: [10.1016/j.pneurobio.2015.09.001](https://doi.org/10.1016/j.pneurobio.2015.09.001)

## Abstract
This paper develops an active inference account of homeostatic regulation and adaptive behavioral control. The authors show how basic homeostatic processes (maintaining physiological set points) and sophisticated goal-directed behavior can both be understood as active inference -- minimizing the discrepancy between predicted and preferred states. The paper bridges autonomic regulation, interoception, motivation, and decision making under a single framework, demonstrating how "desires" and "goals" reduce to prior preferences in generative models.

## Key Contributions

### Homeostasis as Active Inference
- **Set Points as Priors**: Homeostatic set points are prior preferences over interoceptive states
- **Allostasis**: Anticipatory regulation through predictive modeling of future states
- **Interoceptive Inference**: The body's internal state as a target of inference
- **Autonomic Regulation**: Reflexes as simple active inference

### From Homeostasis to Goal-Directed Behavior
- **Continuum**: Homeostatic regulation and goal-directed behavior differ in complexity, not kind
- **Prior Preferences**: Goals are prior preferences at higher levels of the generative model
- **Motivation**: Drive states arise from deviations between predicted and preferred states
- **Action Selection**: Choosing actions that minimize expected free energy

### Bridging Autonomic and Cognitive
- **Interoception-Exteroception**: Internal and external sensing as parallel inference channels
- **Visceral Predictions**: Brain predicts body states just as it predicts sensory states
- **Embodied Decision Making**: Bodily states influence cognitive decisions
- **Emotional Regulation**: Managing prediction errors across internal and external domains

## Core Concepts

### Homeostatic Set Points as Priors
Traditional homeostasis:
```
Error = Actual_State - Set_Point
Action = -K * Error  (negative feedback)
```

Active inference reformulation:
```
Prior Preference: p(o) = N(set_point, precision)
Free Energy: F includes deviation from prior preferences
Action: a = argmin F (minimize deviation from preferred states)
```

### Allostatic Regulation
Allostasis extends homeostasis through prediction:
```
Allostatic Action: Act now to prevent future deviations
Expected Free Energy: G(pi) includes future deviations from preferences
Anticipatory Regulation: Eating before hunger, warming before cold
```

### Hierarchy of Regulation
1. **Spinal Reflexes**: Simple homeostatic loops (fast, local)
2. **Autonomic Regulation**: Visceral set-point maintenance
3. **Motivated Behavior**: Seeking resources to maintain set points
4. **Goal-Directed Planning**: Complex action sequences for long-term regulation
5. **Social Behavior**: Coordinating with others for collective regulation

## Mathematical Formalism

### Interoceptive Generative Model
```
p(o_int, o_ext, s_int, s_ext) = p(o_int|s_int) * p(o_ext|s_ext) * p(s_int, s_ext)
```

Where:
- `o_int`: Interoceptive observations (body signals)
- `o_ext`: Exteroceptive observations (sensory signals)
- `s_int`: Hidden body states
- `s_ext`: Hidden environmental states

### Motivation as Free Energy
Drive states correspond to free energy from interoceptive deviations:
```
F_drive = E_q[-ln p(o_int|C_int)]
```

Where `C_int` encodes prior preferences for physiological states (temperature, glucose, hydration).

## Neuroscience Connections

### Neural Architecture
- **Hypothalamus**: Core homeostatic regulation
- **Insular Cortex**: Interoceptive inference and body awareness
- **Anterior Cingulate**: Integration of interoceptive and exteroceptive inference
- **Orbitofrontal Cortex**: Valuation and preference encoding
- **Brainstem**: Basic autonomic reflexes as active inference

### Neuromodulatory Systems
- **Dopamine**: Precision over exteroceptive predictions
- **Serotonin**: Precision over interoceptive predictions
- **Norepinephrine**: Arousal and global precision
- **Oxytocin**: Social regulation and affective bonding

## Impact and Applications

### Physiology
- **Unified Account**: Single framework for autonomic and behavioral regulation
- **Allostatic Load**: Chronic free energy as physiological stress
- **Eating Disorders**: Aberrant interoceptive inference
- **Pain**: Interoceptive prediction errors

### Psychology
- **Motivation**: Formal account of drive and incentive
- **Emotion**: Link between body states and affective experience
- **Self-Regulation**: Strategies for managing free energy
- **Addiction**: Aberrant prior preferences and precision

### Clinical
- **Psychosomatic Medicine**: Mind-body interactions through shared inference
- **Eating Disorders**: Distorted interoceptive generative models
- **Anxiety**: Excessive interoceptive prediction errors
- **Alexithymia**: Impaired interoceptive inference

## Related Work

### Foundational Papers
- [[friston_2010]] - Free energy principle review
- [[friston_2013]] - Life as we know it

### Extensions
- [[seth_2021]] - Interoceptive inference and selfhood
- [[hesp_2021]] - Deep active inference and affect
- [[parr_pezzulo_friston_2022]] - Textbook treatment

### Related Concepts
- [[friston_2015_knowing]] - Pattern regulation
- [[ramstead_2018]] - Variational ecology

## Citations and Influence
This paper has been widely influential in connecting the FEP to biological regulation, motivation, and embodied cognition. It provided the foundation for subsequent work on interoceptive inference, embodied selfhood, and computational psychiatry of bodily disorders.

## Reading Guide
1. **Introduction**: From homeostasis to active inference
2. **Homeostatic Control**: Set points as prior preferences
3. **Allostasis**: Predictive regulation
4. **Goal-Directed Behavior**: Complex regulation strategies
5. **Neural Implementation**: Brain systems for homeostatic inference

---

> **Homeostasis Unified**: Shows that basic homeostatic regulation and complex goal-directed behavior are both instances of active inference.

---

> **Prior Preferences as Goals**: Demonstrates how "desires" and "goals" reduce to prior preferences in generative models.

---

> **Embodied Active Inference**: Bridges autonomic regulation, interoception, motivation, and decision making under the free energy principle.
